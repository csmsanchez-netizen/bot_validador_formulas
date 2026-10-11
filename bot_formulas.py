"""
Bot de Discord: apoyo a la validación de fórmulas de premezclas.

Flujo:
  1. Alguien escribe "$formula" y adjunta una foto, imagen o PDF (o pega el texto).
  2. La IA transcribe la fórmula a una tabla (insumo / porcentaje).
  3. El bot muestra la tabla y una persona confirma con ✅ (o rechaza con ❌).
  4. Reglas en código (sin IA) revisan: suma 100 %, carga higroscópica,
     insumos sin datos y pares de riesgo (p. ej. minerales traza + vitaminas).
  5. La IA redacta el análisis de interacciones, usando SOLO los datos que le damos
     más su conocimiento general, indicando qué es dato y qué es suposición.
  6. Cada consulta queda registrada en registro_consultas.csv.

IMPORTANTE: es una herramienta de apoyo. La decisión de liberar o no una fórmula
es de Calidad / Formulación.
"""

import asyncio
import base64
import csv
import io
import json
import os
import re
import unicodedata
from datetime import datetime

import discord
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()

# ----------------------------------------------------------------------------
# Configuración (todo se puede cambiar desde el archivo .env)
# ----------------------------------------------------------------------------
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
MODELO = os.getenv("OPENAI_MODEL", "gpt-4o")  # debe aceptar imágenes
ROL_CALIDAD_ID = os.getenv("ROL_CALIDAD_ID")  # opcional: ID del rol a mencionar en rojo
CANAL_ID = os.getenv("CANAL_ID")  # opcional: limitar el bot a un canal

ARCHIVO_INSUMOS = "insumos.csv"
ARCHIVO_REGISTRO = "registro_consultas.csv"

# Umbrales de carga higroscópica (% de insumos con higroscopicidad "alta").
# SON VALORES DE PARTIDA. Calibrarlos con el incidente real de aglomeración
# (fórmula con más de 60 % de sulfatos) y con la experiencia de planta.
UMBRAL_AMARILLO = 20.0
UMBRAL_ROJO = 40.0
TOLERANCIA_SUMA = 1.0  # la suma de porcentajes debe estar en 100 ± esto
MAX_BYTES_ARCHIVO = 8 * 1024 * 1024

# Pares de categorías con riesgo conocido. Ajustar con la experiencia de la planta.
PARES_RIESGO = [
    (
        "mineral_traza",
        "vitamina",
        "Minerales traza (sulfatos y óxidos) junto a vitaminas: posible pérdida de "
        "estabilidad de las vitaminas por oxidación. Revisar vitaminas protegidas, "
        "humedad y tiempo de almacenamiento.",
    ),
    (
        "colina",
        "vitamina",
        "Cloruro de colina junto a vitaminas: puede acelerar su degradación y aporta "
        "mucha humedad por ser muy higroscópico.",
    ),
]

client_ai = OpenAI()  # lee OPENAI_API_KEY del .env

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(
    intents=intents,
    allowed_mentions=discord.AllowedMentions(roles=True, users=True, everyone=False),
)


# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------
def normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    texto = re.sub(r"[^a-z0-9 ]+", " ", texto.lower())
    return re.sub(r"\s+", " ", texto).strip()


def cargar_insumos():
    """Lee insumos.csv y devuelve una lista de (nombres_normalizados, fila)."""
    insumos = []
    if not os.path.exists(ARCHIVO_INSUMOS):
        return insumos
    with open(ARCHIVO_INSUMOS, encoding="utf-8-sig", newline="") as f:
        for fila in csv.DictReader(f):
            nombres = {normalizar(fila["nombre"])}
            for alias in (fila.get("aliases") or "").split(";"):
                if alias.strip():
                    nombres.add(normalizar(alias))
            insumos.append((nombres, fila))
    return insumos


def buscar_insumo(nombre: str, base):
    n = normalizar(nombre)
    for nombres, fila in base:
        if n in nombres:
            return fila
    for nombres, fila in base:
        for candidato in nombres:
            if len(candidato) >= 4 and (candidato in n or n in candidato):
                return fila
    return None


def trocear(texto: str, limite: int = 1900):
    """Discord limita los mensajes a 2000 caracteres."""
    partes, actual = [], ""
    for linea in texto.split("\n"):
        if len(actual) + len(linea) + 1 > limite:
            partes.append(actual)
            actual = ""
        actual += linea + "\n"
    if actual.strip():
        partes.append(actual)
    return partes


async def enviar_largo(canal, texto: str):
    for parte in trocear(texto):
        await canal.send(parte)


# ----------------------------------------------------------------------------
# Paso 1: leer el archivo adjunto
# ----------------------------------------------------------------------------
async def leer_adjuntos(message):
    """Devuelve (texto_extraido, lista_de_imagenes, avisos)."""
    texto, imagenes, avisos = "", [], []
    for att in message.attachments:
        nombre = att.filename.lower()
        tipo = att.content_type or ""
        if att.size > MAX_BYTES_ARCHIVO:
            avisos.append(f"`{att.filename}` supera el tamaño máximo permitido.")
            continue
        datos = await att.read()
        if tipo.startswith("image/") or nombre.endswith((".png", ".jpg", ".jpeg", ".webp")):
            mime = tipo if tipo.startswith("image/") else "image/jpeg"
            imagenes.append((mime, base64.b64encode(datos).decode()))
        elif nombre.endswith(".pdf"):
            try:
                lector = PdfReader(io.BytesIO(datos))
                contenido = "\n".join((p.extract_text() or "") for p in lector.pages)
            except Exception:
                contenido = ""
            if len(contenido.strip()) < 30:
                avisos.append(
                    f"`{att.filename}` parece un PDF escaneado (sin texto). "
                    "Envíalo como foto o captura de pantalla."
                )
            else:
                texto += "\n" + contenido
        else:
            avisos.append(f"`{att.filename}`: formato no soportado (usa imagen o PDF).")
    return texto, imagenes, avisos


# ----------------------------------------------------------------------------
# Paso 2: transcribir la fórmula a una tabla con la IA
# ----------------------------------------------------------------------------
PROMPT_EXTRACCION = """Transcribe la fórmula de una premezcla del material entregado.
Devuelve SOLO un JSON con este formato:
{"insumos": [{"nombre": "texto tal como aparece", "porcentaje": numero}],
 "unidad_original": "% | kg | g | otra",
 "notas": "dudas de lectura o datos ilegibles"}
Reglas:
- No inventes ni corrijas valores. Si algo es ilegible, ponlo en "notas" y omite o marca el insumo.
- Usa punto como separador decimal.
- Si las cantidades vienen en kg o g (no en %), conviértelas a porcentaje del total
  y dilo en "unidad_original".
"""


def _extraer_formula_sync(texto: str, imagenes):
    contenido = [{"type": "text", "text": PROMPT_EXTRACCION + (f"\nTEXTO:\n{texto}" if texto else "")}]
    for mime, b64 in imagenes:
        contenido.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
    resp = client_ai.chat.completions.create(
        model=MODELO,
        messages=[{"role": "user", "content": contenido}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(resp.choices[0].message.content)


# ----------------------------------------------------------------------------
# Paso 4: reglas deterministas (sin IA)
# ----------------------------------------------------------------------------
def evaluar_reglas(insumos_formula, base):
    suma = sum(float(i.get("porcentaje") or 0) for i in insumos_formula)
    alertas, desconocidos, categorias = [], [], set()
    higro_alto = higro_media = 0.0
    detalle = []

    for item in insumos_formula:
        pct = float(item.get("porcentaje") or 0)
        fila = buscar_insumo(item.get("nombre", ""), base)
        if fila is None:
            desconocidos.append(item.get("nombre", "?"))
            continue
        categorias.add(fila["categoria"])
        if fila["higroscopicidad"] == "alta":
            higro_alto += pct
        elif fila["higroscopicidad"] == "media":
            higro_media += pct
        detalle.append({**fila, "porcentaje": pct, "nombre_formula": item.get("nombre")})

    semaforo = "VERDE"

    def subir(nivel):
        nonlocal semaforo
        orden = {"VERDE": 0, "AMARILLO": 1, "ROJO": 2}
        if orden[nivel] > orden[semaforo]:
            semaforo = nivel

    if abs(suma - 100) > TOLERANCIA_SUMA:
        alertas.append(f"La suma de porcentajes es {suma:.2f} % (debería ser 100 %). Posible error de lectura o de fórmula.")
        subir("AMARILLO")
    if higro_alto >= UMBRAL_ROJO:
        alertas.append(f"Carga higroscópica ALTA: {higro_alto:.1f} % de insumos muy higroscópicos (umbral rojo {UMBRAL_ROJO:.0f} %). Riesgo alto de apelmazamiento/aglomeración.")
        subir("ROJO")
    elif higro_alto >= UMBRAL_AMARILLO:
        alertas.append(f"Carga higroscópica elevada: {higro_alto:.1f} % de insumos muy higroscópicos (umbral amarillo {UMBRAL_AMARILLO:.0f} %).")
        subir("AMARILLO")
    if desconocidos:
        alertas.append("Insumos SIN datos en la base (no evaluados por las reglas): " + ", ".join(desconocidos))
        subir("AMARILLO")
    for cat_a, cat_b, mensaje in PARES_RIESGO:
        if cat_a in categorias and cat_b in categorias:
            alertas.append(mensaje)
            subir("AMARILLO")

    return {
        "suma": suma,
        "higro_alto_pct": higro_alto,
        "higro_media_pct": higro_media,
        "desconocidos": desconocidos,
        "alertas": alertas,
        "semaforo": semaforo,
        "detalle": detalle,
    }


# ----------------------------------------------------------------------------
# Paso 5: análisis de interacciones con la IA
# ----------------------------------------------------------------------------
PROMPT_ANALISIS = """Eres un asistente de apoyo para el equipo de Calidad y Formulación de una planta
de premezclas veterinarias. Analizas una fórmula YA CONFIRMADA por una persona.

Recibirás: la fórmula, los datos de la base interna de la planta para cada insumo conocido,
y el resultado de reglas automáticas.

Responde en español, con estas secciones:
1. Interacciones físicas posibles (higroscopicidad, apelmazamiento, segregación por
   densidad o tamaño de partícula, cambios de color o aspecto).
2. Interacciones químicas posibles (oxidación, hidrólisis, incompatibilidades, pérdida
   de actividad de vitaminas y otros activos) y cómo podrían afectar el aspecto o la
   estabilidad en el tiempo.
3. Qué insumos o combinaciones pesan más en el riesgo.
4. Acciones sugeridas para que Calidad evalúe (ajustar carga, cambiar la forma del
   insumo, antiaglomerante, orden de adición, empaque, condiciones de almacenamiento).
5. Datos que faltan para concluir.

Reglas estrictas:
- Distingue claramente "dato de la base de la planta" de "conocimiento general (suposición)".
- No inventes cifras ni umbrales. Si no tienes el dato, dilo.
- Si un insumo no está en la base, no asumas sus propiedades: indícalo como incertidumbre.
- No apruebes ni rechaces la fórmula: tú solo alertas. La decisión es de Calidad.
- Sé concreto y breve.
"""


def _analizar_sync(formula, reglas):
    mensaje = {
        "formula": formula,
        "datos_base_planta": [
            {k: d[k] for k in ("nombre_formula", "porcentaje", "categoria", "higroscopicidad", "notas")}
            for d in reglas["detalle"]
        ],
        "resultado_reglas": {
            "suma_pct": reglas["suma"],
            "higroscopicos_altos_pct": reglas["higro_alto_pct"],
            "higroscopicos_medios_pct": reglas["higro_media_pct"],
            "alertas": reglas["alertas"],
            "semaforo": reglas["semaforo"],
        },
    }
    resp = client_ai.chat.completions.create(
        model=MODELO,
        messages=[
            {"role": "system", "content": PROMPT_ANALISIS},
            {"role": "user", "content": json.dumps(mensaje, ensure_ascii=False)},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content


# ----------------------------------------------------------------------------
# Registro de consultas (trazabilidad)
# ----------------------------------------------------------------------------
def registrar(usuario, formula, reglas, confirmado):
    nuevo = not os.path.exists(ARCHIVO_REGISTRO)
    with open(ARCHIVO_REGISTRO, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(["fecha_hora", "usuario", "confirmada", "semaforo_reglas", "alertas", "formula_json"])
        w.writerow([
            datetime.now().isoformat(timespec="seconds"),
            usuario,
            "si" if confirmado else "no",
            reglas["semaforo"] if reglas else "",
            " | ".join(reglas["alertas"]) if reglas else "",
            json.dumps(formula, ensure_ascii=False),
        ])


def tabla_texto(insumos):
    filas = [f"{'Insumo':<38}{'%':>8}", "-" * 46]
    for i in insumos:
        filas.append(f"{str(i.get('nombre', '?'))[:37]:<38}{float(i.get('porcentaje') or 0):>8.2f}")
    filas.append("-" * 46)
    filas.append(f"{'TOTAL':<38}{sum(float(i.get('porcentaje') or 0) for i in insumos):>8.2f}")
    return "```\n" + "\n".join(filas) + "\n```"


# ----------------------------------------------------------------------------
# Discord
# ----------------------------------------------------------------------------
@client.event
async def on_ready():
    print(f"Conectado como {client.user}")


@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if CANAL_ID and str(message.channel.id) != CANAL_ID:
        return
    if not message.content.startswith("$formula"):
        return

    texto_manual = message.content[len("$formula"):].strip()
    texto_pdf, imagenes, avisos = await leer_adjuntos(message)
    texto = (texto_manual + "\n" + texto_pdf).strip()

    for aviso in avisos:
        await message.channel.send(aviso)
    if not texto and not imagenes:
        await message.channel.send(
            "Escribe `$formula` y adjunta una foto, imagen o PDF de la fórmula (o pega el texto)."
        )
        return

    # Paso 2: transcripción
    async with message.channel.typing():
        try:
            formula = await asyncio.to_thread(_extraer_formula_sync, texto, imagenes)
        except Exception as e:
            await message.channel.send(f"No pude leer la fórmula: {e}")
            return

    insumos = formula.get("insumos", [])
    if not insumos:
        await message.channel.send("No pude identificar insumos en el material. Prueba con una imagen más clara.")
        return

    # Paso 3: confirmación humana
    resumen = "**Esto es lo que entendí de la fórmula:**\n" + tabla_texto(insumos)
    if formula.get("notas"):
        resumen += f"\nNotas de lectura: {formula['notas']}"
    resumen += "\n¿Está correcta? Reacciona con ✅ para analizar o ❌ para descartarla (5 min)."
    conf = await message.channel.send(resumen)
    await conf.add_reaction("✅")
    await conf.add_reaction("❌")

    def check(reaction, user):
        return (
            reaction.message.id == conf.id
            and user.id == message.author.id
            and str(reaction.emoji) in ("✅", "❌")
        )

    try:
        reaction, _ = await client.wait_for("reaction_add", timeout=300, check=check)
    except asyncio.TimeoutError:
        registrar(str(message.author), formula, None, False)
        await message.channel.send("Sin confirmación. Envía `$formula` de nuevo cuando quieras.")
        return

    if str(reaction.emoji) == "❌":
        registrar(str(message.author), formula, None, False)
        await message.channel.send("Descartada. Corrige el material y vuelve a enviar `$formula`.")
        return

    # Pasos 4 y 5: reglas + análisis
    base = cargar_insumos()
    reglas = evaluar_reglas(insumos, base)
    icono = {"VERDE": "🟢", "AMARILLO": "🟡", "ROJO": "🔴"}[reglas["semaforo"]]

    encabezado = f"{icono} **Semáforo de reglas: {reglas['semaforo']}**\n"
    encabezado += "\n".join(f"• {a}" for a in reglas["alertas"]) or "• Las reglas automáticas no detectaron alertas."
    if reglas["semaforo"] == "ROJO" and ROL_CALIDAD_ID:
        encabezado += f"\n<@&{ROL_CALIDAD_ID}> se requiere revisión de Calidad."
    await enviar_largo(message.channel, encabezado)

    async with message.channel.typing():
        try:
            analisis = await asyncio.to_thread(_analizar_sync, formula, reglas)
        except Exception as e:
            await message.channel.send(f"Las reglas se aplicaron, pero falló el análisis con IA: {e}")
            registrar(str(message.author), formula, reglas, True)
            return

    await enviar_largo(message.channel, "**Análisis de interacciones (apoyo, no es una aprobación):**\n" + analisis)
    await message.channel.send("ℹ️ La decisión final de liberar o ajustar la fórmula es de Calidad / Formulación.")
    registrar(str(message.author), formula, reglas, True)


if not DISCORD_TOKEN:
    raise SystemExit("Falta DISCORD_TOKEN en el archivo .env")
client.run(DISCORD_TOKEN)
