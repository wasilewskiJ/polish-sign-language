#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
SKRYPT DO ZBIERANIA DATASETU - Polish Sign Language (PJM)
═══════════════════════════════════════════════════════════════════════════════

FUNKCJONALNOŚĆ:
- Otwiera kamerę z live preview
- Wykrywa dłoń używając MediaPipe
- Pokazuje landmarks dłoni na żywo
- Dodaje ramkę (+100px) wokół wykrytej dłoni
- Zapisuje zdjęcie po naciśnięciu SPACJI (tylko obszar ramki)
- Automatycznie inkrementuje numer pliku dla każdej litery
- Przełączanie między literami klawiszami A-Z
- Wyjście przez ESC

STEROWANIE:
- SPACJA: Zrób zdjęcie
- A-Z: Przełącz aktualną literę
- ESC: Wyjdź

═══════════════════════════════════════════════════════════════════════════════
"""

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import re

# ═══════════════════════════════════════════════════════════════════════════
# KONFIGURACJA
# ═══════════════════════════════════════════════════════════════════════════

# Ścieżka bazowa do folderu z danymi
BASE_DIR = Path(__file__).parent / "backend" / "translator" / "data" / "raw"

# Padding wokół wykrytej dłoni (w pikselach)
PADDING = 70

# Wszystkie dostępne litery (A-Z bez J - według alfabetu PJM)
LETTERS = [chr(i) for i in range(ord('A'), ord('Z') + 1) if chr(i) != 'J']

# Kolory do interfejsu (BGR format dla OpenCV)
COLOR_GREEN = (0, 255, 0)    # Zielony - ramka wokół dłoni
COLOR_RED = (0, 0, 255)      # Czerwony - tekst ostrzeżenia
COLOR_WHITE = (255, 255, 255)  # Biały - tekst informacyjny
COLOR_BLUE = (255, 0, 0)     # Niebieski - landmarks

# ═══════════════════════════════════════════════════════════════════════════
# INICJALIZACJA MEDIAPIPE
# ═══════════════════════════════════════════════════════════════════════════

# MediaPipe Hands - wykrywanie dłoni i landmarks
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Konfiguracja wykrywania dłoni
hands = mp_hands.Hands(
    static_image_mode=False,      # Tryb video (False = optymalizacja dla wideo)
    max_num_hands=1,               # Wykrywaj maksymalnie 1 dłoń
    min_detection_confidence=0.5,  # Próg pewności detekcji (0-1)
    min_tracking_confidence=0.5    # Próg pewności trackingu (0-1)
)


# ═══════════════════════════════════════════════════════════════════════════
# FUNKCJE POMOCNICZE
# ═══════════════════════════════════════════════════════════════════════════

def get_next_index(letter):
    """
    Znajduje następny dostępny indeks dla danej litery.
    
    PRZYKŁAD:
        Jeśli w folderze B są pliki: B1.jpg, B2.jpg, B5.jpg, B26.jpg
        To zwróci: 27 (maksymalny + 1)
    
    PARAMETRY:
        letter - litera (np. 'A', 'B', ...)
    
    ZWRACA:
        int - następny indeks do użycia
    """
    # Ścieżka do folderu z literą
    letter_dir = BASE_DIR / letter
    
    # Upewnij się że folder istnieje
    letter_dir.mkdir(parents=True, exist_ok=True)
    
    # Wzorzec pliku: {LETTER}{NUMER}.jpg (np. A1.jpg, B26.jpg)
    # Regex: nazwa_litery + cyfry + .jpg
    pattern = re.compile(rf"{letter}(\d+)\.jpg")
    
    # Lista znalezionych indeksów
    indices = []
    
    # Przeszukaj pliki w folderze
    for file_path in letter_dir.glob("*.jpg"):
        match = pattern.match(file_path.name)
        if match:
            # Wyciągnij numer z nazwy pliku
            indices.append(int(match.group(1)))
    
    # Jeśli są jakieś pliki -> zwróć max + 1, w przeciwnym razie 1
    return max(indices) + 1 if indices else 1


def get_hand_bbox(hand_landmarks, image_width, image_height, padding=PADDING):
    """
    Oblicza bounding box wokół wykrytej dłoni z paddingiem.
    
    PARAMETRY:
        hand_landmarks - wykryte landmarks dłoni (MediaPipe)
        image_width - szerokość obrazu
        image_height - wysokość obrazu
        padding - padding wokół dłoni (w pikselach)
    
    ZWRACA:
        (x_min, y_min, x_max, y_max) - współrzędne bounding boxa
        lub None jeśli nie można obliczyć
    """
    if not hand_landmarks:
        return None
    
    # Zbierz wszystkie współrzędne x i y z landmarks
    x_coords = [lm.x * image_width for lm in hand_landmarks.landmark]
    y_coords = [lm.y * image_height for lm in hand_landmarks.landmark]
    
    # Znajdź min i max
    x_min = int(min(x_coords))
    y_min = int(min(y_coords))
    x_max = int(max(x_coords))
    y_max = int(max(y_coords))
    
    # Dodaj padding
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(image_width, x_max + padding)
    y_max = min(image_height, y_max + padding)
    
    return (x_min, y_min, x_max, y_max)


def save_cropped_image(frame, bbox, letter):
    """
    Zapisuje przycięty obraz (tylko obszar dłoni) do pliku.
    
    WAŻNE: frame powinien być ORYGINALNY (bez landmarks i ramki)!
           Landmarks i ramka są tylko do wyświetlania na ekranie.
    
    PARAMETRY:
        frame - CZYSTY obraz z kamery (bez rysunków!)
        bbox - bounding box (x_min, y_min, x_max, y_max)
        letter - aktualna litera (A-Z)
    
    ZWRACA:
        str - ścieżka do zapisanego pliku lub None jeśli błąd
    """
    if bbox is None:
        return None
    
    x_min, y_min, x_max, y_max = bbox
    
    # Wytnij obszar dłoni
    cropped = frame[y_min:y_max, x_min:x_max]
    
    # Jeśli obszar jest pusty - return None
    if cropped.size == 0:
        return None
    
    # Znajdź następny indeks
    next_idx = get_next_index(letter)
    
    # Ścieżka do zapisu
    output_path = BASE_DIR / letter / f"{letter}{next_idx}.jpg"
    
    # Zapisz obraz
    cv2.imwrite(str(output_path), cropped)
    
    return str(output_path)


def draw_ui(frame, current_letter, hand_detected, last_saved):
    """
    Rysuje interfejs użytkownika na obrazie.
    
    PARAMETRY:
        frame - obraz z kamery
        current_letter - aktualna litera
        hand_detected - czy wykryto dłoń (True/False)
        last_saved - ścieżka do ostatnio zapisanego zdjęcia (lub None)
    """
    height, width = frame.shape[:2]
    
    # ─────────────────────────────────────────────────────────────────────
    # Panel informacyjny (na górze)
    # ─────────────────────────────────────────────────────────────────────
    
    # Tło dla tekstu (półprzezroczyste)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    # Aktualna litera (duży tekst)
    cv2.putText(
        frame,
        f"Litera: {current_letter}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        COLOR_WHITE,
        2
    )
    
    # Następny indeks
    next_idx = get_next_index(current_letter)
    cv2.putText(
        frame,
        f"Nastepny: {current_letter}{next_idx}.jpg",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        COLOR_WHITE,
        1
    )
    
    # Status dłoni
    status_text = "Dlon: WYKRYTA" if hand_detected else "Dlon: BRAK"
    status_color = COLOR_GREEN if hand_detected else COLOR_RED
    cv2.putText(
        frame,
        status_text,
        (300, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_color,
        2
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # Panel z ostatnio zapisanym zdjęciem
    # ─────────────────────────────────────────────────────────────────────
    if last_saved:
        cv2.putText(
            frame,
            f"Zapisano: {Path(last_saved).name}",
            (300, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLOR_GREEN,
            1
        )
    
    # ─────────────────────────────────────────────────────────────────────
    # Instrukcje (na dole)
    # ─────────────────────────────────────────────────────────────────────
    
    # Tło
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, height - 80), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    # Tekst instrukcji
    cv2.putText(
        frame,
        "SPACJA: Zrob zdjecie  |  A-Z: Zmien litere  |  ESC: Wyjdz",
        (10, height - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        COLOR_WHITE,
        1
    )
    
    # Statystyka zdjęć
    total_images = sum(len(list((BASE_DIR / letter).glob("*.jpg"))) for letter in LETTERS if (BASE_DIR / letter).exists())
    cv2.putText(
        frame,
        f"Calkowita liczba zdjec: {total_images}",
        (10, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        COLOR_WHITE,
        1
    )


# ═══════════════════════════════════════════════════════════════════════════
# GŁÓWNA PĘTLA APLIKACJI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """
    Główna funkcja - uruchamia aplikację do zbierania datasetu.
    """
    # ─────────────────────────────────────────────────────────────────────
    # INICJALIZACJA
    # ─────────────────────────────────────────────────────────────────────
    
    print("═" * 80)
    print("ZBIERANIE DATASETU - Polish Sign Language")
    print("═" * 80)
    print()
    print("Otwieranie kamery...")
    
    # Otwórz kamerę (0 = domyślna kamera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ BŁĄD: Nie można otworzyć kamery!")
        return
    
    print("✅ Kamera otwarta!")
    print()
    print("STEROWANIE:")
    print("  SPACJA: Zrób zdjęcie")
    print("  A-Z: Przełącz literę")
    print("  ESC: Wyjdź")
    print()
    print("═" * 80)
    
    # Aktualna litera (start od A)
    current_letter = 'A'
    
    # Ostatnio zapisane zdjęcie (dla UI)
    last_saved = None
    
    # Licznik zapisanych zdjęć (dla feedbacku)
    saved_count = 0
    
    # ─────────────────────────────────────────────────────────────────────
    # GŁÓWNA PĘTLA
    # ─────────────────────────────────────────────────────────────────────
    
    while True:
        # Wczytaj klatkę z kamery
        ret, frame = cap.read()
        
        if not ret:
            print("❌ BŁĄD: Nie można odczytać klatki z kamery!")
            break
        
        # Odbij obraz w poziomie (mirror effect - bardziej intuicyjne)
        frame = cv2.flip(frame, 1)
        
        # Wymiary obrazu
        height, width, _ = frame.shape
        
        # ─────────────────────────────────────────────────────────────────
        # WYKRYWANIE DŁONI (MediaPipe)
        # ─────────────────────────────────────────────────────────────────
        
        # Konwertuj BGR -> RGB (MediaPipe wymaga RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Przetwórz obraz przez MediaPipe
        results = hands.process(rgb_frame)
        
        # Czy wykryto dłoń?
        hand_detected = results.multi_hand_landmarks is not None
        
        # Bounding box wokół dłoni (jeśli wykryto)
        bbox = None
        
        # ─────────────────────────────────────────────────────────────────
        # KOPIA DO WYŚWIETLANIA (z landmarks i ramką)
        # Oryginalny frame zostaje czysty (bez rysunków) - do zapisu!
        # ─────────────────────────────────────────────────────────────────
        display_frame = frame.copy()
        
        if hand_detected:
            # Weź pierwszą wykrytą dłoń (max_num_hands=1, więc zawsze jedna)
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # ─────────────────────────────────────────────────────────────
            # RYSOWANIE LANDMARKS na obrazie DO WYŚWIETLANIA
            # (NIE na oryginalnym frame!)
            # ─────────────────────────────────────────────────────────────
            mp_drawing.draw_landmarks(
                display_frame,  # Rysuj na kopii!
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # ─────────────────────────────────────────────────────────────
            # BOUNDING BOX wokół dłoni
            # ─────────────────────────────────────────────────────────────
            bbox = get_hand_bbox(hand_landmarks, width, height, PADDING)
            
            if bbox:
                x_min, y_min, x_max, y_max = bbox
                
                # Rysuj zieloną ramkę wokół dłoni (na kopii do wyświetlania)
                cv2.rectangle(
                    display_frame,  # Rysuj na kopii!
                    (x_min, y_min),
                    (x_max, y_max),
                    COLOR_GREEN,
                    2
                )
                
                # Informacja o rozmiarze wyciętego obrazu (na kopii)
                crop_width = x_max - x_min
                crop_height = y_max - y_min
                cv2.putText(
                    display_frame,  # Rysuj na kopii!
                    f"{crop_width}x{crop_height}px",
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    COLOR_GREEN,
                    1
                )
        
        # ─────────────────────────────────────────────────────────────────
        # INTERFEJS UŻYTKOWNIKA (rysuj na display_frame)
        # ─────────────────────────────────────────────────────────────────
        draw_ui(display_frame, current_letter, hand_detected, last_saved)
        
        # ─────────────────────────────────────────────────────────────────
        # WYŚWIETL OBRAZ (display_frame z wszystkimi dodatkami)
        # ─────────────────────────────────────────────────────────────────
        cv2.imshow('Zbieranie Datasetu - PJM', display_frame)
        
        # ─────────────────────────────────────────────────────────────────
        # OBSŁUGA KLAWIATURY
        # ─────────────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        
        # ESC - wyjdź
        if key == 27:  # ESC key
            print("\n👋 Zamykanie aplikacji...")
            break
        
        # SPACJA - zrób zdjęcie
        elif key == 32:  # SPACE key
            if bbox is not None:
                # Zapisz przycięte zdjęcie
                saved_path = save_cropped_image(frame, bbox, current_letter)
                
                if saved_path:
                    saved_count += 1
                    last_saved = saved_path
                    print(f"✅ Zapisano: {Path(saved_path).name} (total: {saved_count})")
                else:
                    print("❌ Błąd: Nie można zapisać zdjęcia!")
            else:
                print("⚠️  Brak wykrytej dłoni! Nie można zrobić zdjęcia.")
        
        # A-Z - przełącz literę
        elif 65 <= key <= 90 or 97 <= key <= 122:  # A-Z lub a-z
            new_letter = chr(key).upper()
            
            # Sprawdź czy litera jest dostępna (bez J)
            if new_letter in LETTERS:
                current_letter = new_letter
                last_saved = None  # Reset ostatnio zapisanego
                print(f"📝 Przełączono na literę: {current_letter}")
            else:
                print(f"⚠️  Litera {new_letter} nie jest dostępna!")
    
    # ─────────────────────────────────────────────────────────────────────
    # SPRZĄTANIE
    # ─────────────────────────────────────────────────────────────────────
    
    print()
    print("═" * 80)
    print(f"📊 PODSUMOWANIE:")
    print(f"   Zapisano zdjęć w tej sesji: {saved_count}")
    
    # Statystyka per litera
    print(f"\n📈 Liczba zdjęć per litera:")
    for letter in LETTERS:
        letter_dir = BASE_DIR / letter
        if letter_dir.exists():
            count = len(list(letter_dir.glob("*.jpg")))
            print(f"   {letter}: {count} zdjęć")
    
    print("═" * 80)
    
    # Zwolnij zasoby
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    print("\n✅ Aplikacja zamknięta pomyślnie!")


# ═══════════════════════════════════════════════════════════════════════════
# URUCHOMIENIE
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Przerwano przez użytkownika (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ BŁĄD: {e}")
        import traceback
        traceback.print_exc()
