import cv2
import mediapipe as mp
import time
import math
import numpy as np
import os
import ctypes

# ---------------------------------------------------------
# Resolución Nativa (DPI Awareness)
# ---------------------------------------------------------
try:
    # Asegurar que el sistema operativo no difumine la ventana
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

user32 = ctypes.windll.user32
ANCHO_MONITOR = user32.GetSystemMetrics(0)
ALTO_MONITOR = user32.GetSystemMetrics(1)

# Prevención de fallos en la detección del monitor
if ANCHO_MONITOR < 800 or ALTO_MONITOR < 600:
    ANCHO_MONITOR = 1920
    ALTO_MONITOR = 1200

# ---------------------------------------------------------
# Configuración Inicial y Rutas
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ruta_imagen = os.path.join(BASE_DIR, 'imagen_fondo.jpg')
if os.path.exists(ruta_imagen):
    fondo_usuario = cv2.imread(ruta_imagen)
else:
    # Generación de fondo degradado por defecto
    fondo_usuario = np.zeros((1080, 1920, 3), dtype=np.uint8)
    for i in range(1080):
        fondo_usuario[i, :, 0] = int((i / 1080.0) * 255)
        fondo_usuario[i, :, 1] = int(255 - (i / 1080.0) * 255)
        fondo_usuario[i, :, 2] = 150
        
fondo_redimensionado = False
imagen_secreta = None

# ---------------------------------------------------------
# Inicialización de MediaPipe (Tracking de Manos y Segmentación)
# ---------------------------------------------------------
OpcionesBase = mp.tasks.BaseOptions
DetectorManos = mp.tasks.vision.HandLandmarker
OpcionesDetector = mp.tasks.vision.HandLandmarkerOptions
ModoVideo = mp.tasks.vision.RunningMode

opciones_manos = OpcionesDetector(
    base_options=OpcionesBase(model_asset_path=os.path.join(BASE_DIR, 'hand_landmarker.task')),
    running_mode=ModoVideo.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
reconocedor_manos = DetectorManos.create_from_options(opciones_manos)

segmentador_fondo = None
try:
    SegmentadorImagen = mp.tasks.vision.ImageSegmenter
    OpcionesSegmentador = mp.tasks.vision.ImageSegmenterOptions
    opciones_seg = OpcionesSegmentador(
        base_options=OpcionesBase(model_asset_path=os.path.join(BASE_DIR, 'selfie_segmenter.tflite')),
        running_mode=ModoVideo.VIDEO,
        output_category_mask=True
    )
    segmentador_fondo = SegmentadorImagen.create_from_options(opciones_seg)
except Exception as e:
    print("Advertencia: No se pudo cargar el segmentador de fondo.", e)

# ---------------------------------------------------------
# Configuración de Captura de Video
# ---------------------------------------------------------
camara = cv2.VideoCapture(0)
camara.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
camara.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
camara.set(cv2.CAP_PROP_FPS, 60)

# ---------------------------------------------------------
# Definición de Filtros y Efectos
# ---------------------------------------------------------
lista_filtros = [
    "BLANCO Y NEGRO", 
    "DOS TONOS", 
    "MUNDO INVERTIDO", 
    "VIEJO (SEPIA)", 
    "BORROSO", 
    "DEPREDADOR (TERMICO)", 
    "DIBUJO", 
    "GLITCH LOCO", 
    "MODO HACKER",
    "-F(X)",
    "MINECRAFT (PIXEL)", 
    "IMAGEN SECRETA"
]
filtro_actual = 0
estado_cambio_filtro = False

def aplicar_filtro(roi, nombre_efecto, x_offset=0, y_offset=0, mascara_fondo=None):
    """ Aplica el algoritmo de procesamiento de imagen a la región delimitada. """
    if nombre_efecto == "BLANCO Y NEGRO":
        gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)
        
    elif nombre_efecto == "MUNDO INVERTIDO":
        return cv2.bitwise_not(roi)
        
    elif nombre_efecto == "BORROSO":
        return cv2.GaussianBlur(roi, (25, 25), 0)
        
    elif nombre_efecto == "VIEJO (SEPIA)":
        matriz_sepia = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
        con_filtro = cv2.transform(roi, matriz_sepia)
        return np.clip(con_filtro, 0, 255).astype(np.uint8)
        
    elif nombre_efecto == "DOS TONOS":
        gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mascara_bn = cv2.threshold(gris, 127, 255, cv2.THRESH_BINARY)
        imagen_filtrada = np.zeros_like(roi)
        imagen_filtrada[mascara_bn == 255] = [0, 165, 255]
        imagen_filtrada[mascara_bn == 0] = [147, 20, 255]
        return imagen_filtrada
        
    elif nombre_efecto == "MINECRAFT (PIXEL)":
        alto, ancho = roi.shape[:2]
        if alto > 10 and ancho > 10:
            roi_reducido = cv2.resize(roi, (ancho//10, alto//10), interpolation=cv2.INTER_LINEAR)
            return cv2.resize(roi_reducido, (ancho, alto), interpolation=cv2.INTER_NEAREST)
            
    elif nombre_efecto == "DEPREDADOR (TERMICO)":
        return cv2.applyColorMap(roi, cv2.COLORMAP_JET)
        
    elif nombre_efecto == "DIBUJO":
        gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        invertido = cv2.bitwise_not(gris)
        difuminado = cv2.GaussianBlur(invertido, (21, 21), 0)
        boceto = cv2.divide(gris, 255 - difuminado, scale=256)
        return cv2.cvtColor(boceto, cv2.COLOR_GRAY2BGR)
        
    elif nombre_efecto == "GLITCH LOCO":
        alto, ancho = roi.shape[:2]
        desplazamiento = max(5, ancho // 20)
        imagen_glitch = roi.copy()
        if ancho > desplazamiento:
            imagen_glitch[:, :-desplazamiento, 2] = roi[:, desplazamiento:, 2]
            imagen_glitch[:, desplazamiento:, 0] = roi[:, :-desplazamiento, 0]
        return imagen_glitch
        
    elif nombre_efecto == "IMAGEN SECRETA":
        global imagen_secreta
        alto_roi, ancho_roi = roi.shape[:2]
        y2 = min(imagen_secreta.shape[0], y_offset + alto_roi)
        x2 = min(imagen_secreta.shape[1], x_offset + ancho_roi)
        recorte = imagen_secreta[y_offset : y2, x_offset : x2]
        if recorte.shape[:2] != roi.shape[:2]:
            return roi 
        return recorte.copy()
        
    elif nombre_efecto == "MODO HACKER":
        gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        bordes = cv2.Canny(gris, 60, 120)
        fondo_negro = np.zeros_like(roi)
        fondo_negro[bordes > 0] = [0, 255, 0]
        return fondo_negro
        
    elif nombre_efecto == "-F(X)":
        return cv2.flip(roi, 0)
        
    return roi

# ---------------------------------------------------------
# Sistema de Interfaz Gráfica (Glassmorphism Profesional)
# ---------------------------------------------------------
def renderizar_panel_cristal(img, x, y, w, h, r=24, alpha=0.25):
    """
    Renderiza un panel translúcido con desenfoque de alto rendimiento (Fast-Blur)
    y bordes reactivos que refractan dinámicamente los objetos del fondo.
    """
    if x < 0 or y < 0 or x + w > img.shape[1] or y + h > img.shape[0]: 
        return
        
    roi = img[y:y+h, x:x+w].copy()
    
    # 1. Desenfoque Gaussiano de Alto Rendimiento (Fast-Blur)
    # Reduce drásticamente la latencia de la cámara trabajando en una sub-escala
    roi_pequeno = cv2.resize(roi, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_LINEAR)
    blur_pequeno = cv2.GaussianBlur(roi_pequeno, (15, 15), 0)
    blur_roi = cv2.resize(blur_pequeno, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Superposición clara para el efecto esmerilado
    overlay = np.full((h, w, 3), (245, 245, 250), dtype=np.uint8)
    cristal = cv2.addWeighted(blur_roi, 1.0 - alpha, overlay, alpha, 0)
    
    # Iluminación volumétrica 
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    brillo = np.clip(255 - (X + Y) * 0.4, 0, 255)
    factor_brillo = (brillo / 255.0 * 0.25)[:, :, np.newaxis] 
    cristal = (cristal * (1 - factor_brillo) + 255 * factor_brillo).astype(np.uint8)
    
    # 2. Máscara del panel 
    mascara = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mascara, (r, 0), (w - r, h), 255, -1)
    cv2.rectangle(mascara, (0, r), (w, h - r), 255, -1)
    cv2.circle(mascara, (r, r), r, 255, -1)
    cv2.circle(mascara, (w - r, r), r, 255, -1)
    cv2.circle(mascara, (w - r, h - r), r, 255, -1)
    cv2.circle(mascara, (r, h - r), r, 255, -1)
    
    mascara_3d = mascara[:, :, np.newaxis] / 255.0
    panel_final = (roi * (1 - mascara_3d) + cristal * mascara_3d).astype(np.uint8)
    
    # Simplemente aplicamos el panel final de cristal esmerilado sin bordes extra
    img[y:y+h, x:x+w] = panel_final

def dibujar_texto_sombreado(img, texto, posicion, fuente, escala, color_texto, grosor=1):
    """ Renderizado de texto de alta legibilidad vectorial. """
    x, y = posicion
    color_sombra = (30, 30, 30)
    cv2.putText(img, texto, (x + 2, y + 2), fuente, escala, color_sombra, grosor, cv2.LINE_AA)
    cv2.putText(img, texto, (x, y), fuente, escala, color_texto, grosor, cv2.LINE_AA)

def renderizar_ui_apple(img, ancho_img, alto_img, filtro_actual_str):
    """
    Construye la UI basada en resolución nativa, calculada directamente sobre 
    las dimensiones del lienzo (Monitor).
    """
    # 1. Dynamic Island (Barra superior central)
    texto_isla = f"MODO: {filtro_actual_str}"
    tam_texto_isla = cv2.getTextSize(texto_isla, cv2.FONT_HERSHEY_DUPLEX, 0.85, 2)[0]
    w_isla = tam_texto_isla[0] + 100
    h_isla = 60
    x_isla = (ancho_img - w_isla) // 2
    y_isla = 20
    
    renderizar_panel_cristal(img, x_isla, y_isla, w_isla, h_isla, r=30, alpha=0.3)
    
    radio_punto = 6 if int(time.time() * 1.5) % 2 == 0 else 4
    cv2.circle(img, (x_isla + 30, y_isla + h_isla // 2), radio_punto, (0, 230, 0), cv2.FILLED, cv2.LINE_AA)
    
    dibujar_texto_sombreado(img, texto_isla, (x_isla + 55, y_isla + 38), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2)

    # 2. Panel Inferior Izquierdo (Instrucciones de Gestos)
    texto_gestos_titulo = "Gestos (Cambio de Modo):"
    texto_gestos_1 = "-> Juntar dedos Indices"
    texto_gestos_2 = "-> Juntar Pulgar y Menique"
    
    w_gestos = 440
    h_gestos = 115
    x_gestos = 25
    y_gestos = alto_img - h_gestos - 25
    
    renderizar_panel_cristal(img, x_gestos, y_gestos, w_gestos, h_gestos, r=22, alpha=0.25)
    dibujar_texto_sombreado(img, texto_gestos_titulo, (x_gestos + 25, y_gestos + 35), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)
    dibujar_texto_sombreado(img, texto_gestos_1, (x_gestos + 25, y_gestos + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (230, 230, 230), 2)
    dibujar_texto_sombreado(img, texto_gestos_2, (x_gestos + 25, y_gestos + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (230, 230, 230), 2)

    # 3. Panel Inferior Derecho (Firma de Autor)
    texto_firma = "By Grostar"
    tam_texto_firma = cv2.getTextSize(texto_firma, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)[0]
    w_firma = tam_texto_firma[0] + 50
    h_firma = 54
    x_firma = ancho_img - w_firma - 25
    y_firma = alto_img - h_firma - 25
    
    renderizar_panel_cristal(img, x_firma, y_firma, w_firma, h_firma, r=26, alpha=0.25)
    dibujar_texto_sombreado(img, texto_firma, (x_firma + 25, y_firma + 36), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 220, 255), 2)

    # 4. Panel Superior Izquierdo (Botón de Salida)
    texto_salir = "ESC o Q : Salir"
    tam_texto_salir = cv2.getTextSize(texto_salir, cv2.FONT_HERSHEY_DUPLEX, 0.70, 2)[0]
    w_salir = tam_texto_salir[0] + 50
    h_salir = 54
    x_salir = 25
    y_salir = 20
    
    renderizar_panel_cristal(img, x_salir, y_salir, w_salir, h_salir, r=26, alpha=0.25)
    dibujar_texto_sombreado(img, texto_salir, (x_salir + 25, y_salir + 36), cv2.FONT_HERSHEY_DUPLEX, 0.70, (200, 200, 255), 2)


# ---------------------------------------------------------
# Bucle Principal de Renderizado Nativo
# ---------------------------------------------------------
cv2.namedWindow('Sistema Avanzado de Vision Computacional', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.setWindowProperty('Sistema Avanzado de Vision Computacional', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    exito, cuadro = camara.read()
    if not exito:
        print("Error: No se pudo obtener el fotograma de la cámara.")
        break

    # Efecto espejo
    cuadro = cv2.flip(cuadro, 1) 
    alto_orig, ancho_orig = cuadro.shape[:2]
    
    # Procesamiento algorítmico ligero
    cuadro_rgb = cv2.cvtColor(cuadro, cv2.COLOR_BGR2RGB)
    imagen_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=cuadro_rgb)
    tiempo_ms = time.time_ns() // 1_000_000
    
    resultados_manos = reconocedor_manos.detect_for_video(imagen_mp, tiempo_ms)
    
    # --- CONSTRUCCIÓN DEL LIENZO A LA RESOLUCIÓN NATIVA DEL MONITOR ---
    # Esto asegura que el texto se dibuje con los píxeles reales del monitor,
    # eliminando al 100% el estiramiento y difuminado (Letterbox algorítmico).
    escala = min(ANCHO_MONITOR / ancho_orig, ALTO_MONITOR / alto_orig)
    nuevo_ancho = int(ancho_orig * escala)
    nuevo_alto = int(alto_orig * escala)
    
    cuadro_escalado = cv2.resize(cuadro, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_LINEAR)
    lienzo = np.zeros((ALTO_MONITOR, ANCHO_MONITOR, 3), dtype=np.uint8)
    
    x_offset_l = (ANCHO_MONITOR - nuevo_ancho) // 2
    y_offset_l = (ALTO_MONITOR - nuevo_alto) // 2
    
    lienzo[y_offset_l:y_offset_l+nuevo_alto, x_offset_l:x_offset_l+nuevo_ancho] = cuadro_escalado
    
    # Renderizado adaptativo de la imagen de fondo secreto
    if not fondo_redimensionado:
        imagen_secreta = cv2.resize(fondo_usuario, (ANCHO_MONITOR, ALTO_MONITOR))
        fondo_redimensionado = True

    filtro_elegido = lista_filtros[filtro_actual]
    
    # Pipeline de Segmentación adaptativa (Solo se ejecuta si es necesario para evitar lag)
    mascara_persona = None
    if segmentador_fondo is not None and filtro_elegido == "IMAGEN SECRETA":
        resultado_seg = segmentador_fondo.segment_for_video(imagen_mp, tiempo_ms)
        if resultado_seg.category_mask is not None:
            masc_orig = resultado_seg.category_mask.numpy_view()
            masc_redim = cv2.resize(masc_orig, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_NEAREST)
            mascara_persona = np.zeros((ALTO_MONITOR, ANCHO_MONITOR), dtype=np.uint8)
            mascara_persona[y_offset_l:y_offset_l+nuevo_alto, x_offset_l:x_offset_l+nuevo_ancho] = masc_redim

    puntos_portal = []
    solicitar_cambio_filtro = False

    if resultados_manos.hand_landmarks:
        def mapear_coordenadas(lm):
            """ Traslación vectorial al espacio de resolución nativa del monitor. """
            return int(lm.x * nuevo_ancho) + x_offset_l, int(lm.y * nuevo_alto) + y_offset_l

        # Verificación lógica de colisión (Gesto 1)
        if len(resultados_manos.hand_landmarks) >= 2:
            p1 = mapear_coordenadas(resultados_manos.hand_landmarks[0][8])
            p2 = mapear_coordenadas(resultados_manos.hand_landmarks[1][8])
            if math.hypot(p1[0] - p2[0], p1[1] - p2[1]) < (40 * escala):
                solicitar_cambio_filtro = True
                
        # Verificación lógica de colisión (Gesto 2)
        for puntos_mano in resultados_manos.hand_landmarks:
            p_pulgar = mapear_coordenadas(puntos_mano[4])
            p_menique = mapear_coordenadas(puntos_mano[20])
                
            if math.hypot(p_pulgar[0] - p_menique[0], p_pulgar[1] - p_menique[1]) < (40 * escala):
                solicitar_cambio_filtro = True
                
        if solicitar_cambio_filtro:
            if not estado_cambio_filtro:
                filtro_actual = (filtro_actual + 1) % len(lista_filtros)
                estado_cambio_filtro = True
        else:
            estado_cambio_filtro = False

        # Interfaz cinemática para el tracking
        for puntos_mano in resultados_manos.hand_landmarks:
            for id_dedo, marca in enumerate(puntos_mano):
                coord_x, coord_y = mapear_coordenadas(marca)
                if id_dedo in [4, 8]:
                    puntos_portal.append([coord_x, coord_y])
                    cv2.circle(lienzo, (coord_x, coord_y), max(2, int(6 * escala)), (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.circle(lienzo, (coord_x, coord_y), max(2, int(4 * escala)), (0, 150, 255), cv2.FILLED, cv2.LINE_AA)

        # Generación de geometría del portal
        if len(puntos_portal) == 4:
            puntos_portal.sort(key=lambda p: p[1]) 
            puntos_superior = puntos_portal[:2]
            puntos_inferior = puntos_portal[2:]
            puntos_superior.sort(key=lambda p: p[0]) 
            puntos_inferior.sort(key=lambda p: p[0])
            
            poligono_region = np.array([puntos_superior[0], puntos_superior[1], puntos_inferior[1], puntos_inferior[0]], dtype=np.int32)
            
            x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(poligono_region)
            x_rect, y_rect = max(0, x_rect), max(0, y_rect)
            w_rect = min(ANCHO_MONITOR - x_rect, w_rect)
            h_rect = min(ALTO_MONITOR - y_rect, h_rect)
            
            if w_rect > 0 and h_rect > 0:
                roi = lienzo[y_rect:y_rect+h_rect, x_rect:x_rect+w_rect].copy()
                roi_procesada = aplicar_filtro(roi, filtro_elegido, x_rect, y_rect, mascara_persona)
                
                mascara_poligonal = np.zeros((h_rect, w_rect), dtype=np.uint8)
                poligono_relativo = poligono_region - [x_rect, y_rect]
                cv2.fillPoly(mascara_poligonal, [poligono_relativo], 255)
                mascara_color = cv2.cvtColor(mascara_poligonal, cv2.COLOR_GRAY2BGR)
                
                lienzo[y_rect:y_rect+h_rect, x_rect:x_rect+w_rect] = np.where(mascara_color == 255, roi_procesada, roi)
                cv2.polylines(lienzo, [poligono_region], True, (255, 255, 255), max(1, int(1.5*escala)), cv2.LINE_AA)

    # Renderizado vectorial de los elementos de texto e interfaz en resolución máxima
    renderizar_ui_apple(lienzo, ANCHO_MONITOR, ALTO_MONITOR, filtro_elegido)

    # Despliegue en ventana
    cv2.imshow('Sistema Avanzado de Vision Computacional', lienzo)
    
    # Captura de eventos del teclado
    tecla = cv2.waitKey(1) & 0xFF
    if tecla == ord('q') or tecla == ord('Q') or tecla == 27:
        break

# ---------------------------------------------------------
# Liberación de Recursos de Hardware
# ---------------------------------------------------------
camara.release()
cv2.destroyAllWindows()