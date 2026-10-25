import csv
import json
import math
import time
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import yt_dlp
from ultralytics import YOLO


# ============================================================
# USTAWIENIA
# ============================================================

YOUTUBE_URL = "https://www.youtube.com/watch?v=Q71sLS8h9a4"
MODEL = "yolov8n.pt"

# Maksymalna liczba wyświetlanych i analizowanych klatek/s.
# Klatki nie są celowo pomijane.
TARGET_FPS = 15.0

# Wartość zastępcza, gdy nie można odczytać FPS źródła.
FALLBACK_FPS = 25.0

FRAME_WIDTH = 1280

# Postój i umowne punkty.
MIN_STOP_SECONDS = 5
SECONDS_PER_POINT = 5
MOVEMENT_TOLERANCE = 8
MAX_MISSING_SECONDS = 2

BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "strefy.json"
OUTPUT_DIR = BASE_DIR / "wyniki"

PARKING_COLOR = (0, 220, 255)
MEASUREMENT_COLOR = (255, 170, 0)
HEADER_HEIGHT = 100


# ============================================================
# OBSŁUGA PLIKÓW
# ============================================================

def write_json(path, data):
    temporary = path.with_suffix(".tmp")

    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    temporary.replace(path)


def valid_rect(rect):
    return (
        isinstance(rect, list)
        and len(rect) == 4
        and all(isinstance(value, (int, float)) for value in rect)
        and 0 <= rect[0] < rect[2] <= 1
        and 0 <= rect[1] < rect[3] <= 1
    )


def load_config():
    if not CONFIG_FILE.exists():
        return {
            "parking": None,
            "measurements": [],
        }

    data = json.loads(
        CONFIG_FILE.read_text(encoding="utf-8")
    )

    if not isinstance(data, dict):
        raise ValueError("Niepoprawny plik strefy.json.")

    if data.get("parking") is not None:
        if not valid_rect(data["parking"]):
            raise ValueError("Niepoprawna strefa postoju.")

    zones = data.get("measurements")

    if not isinstance(zones, list):
        raise ValueError("Niepoprawna lista stref pomiarowych.")

    used_ids = set()

    for zone in zones:
        if (
            not isinstance(zone, dict)
            or not isinstance(zone.get("id"), str)
            or not isinstance(zone.get("name"), str)
            or not valid_rect(zone.get("rect"))
            or zone["id"] in used_ids
        ):
            raise ValueError("Niepoprawna strefa pomiarowa.")

        used_ids.add(zone["id"])

    return data


# ============================================================
# STRUMIEŃ I GEOMETRIA
# ============================================================

def get_stream_url():
    options = {
        "format": "best[height<=720]/bestvideo[height<=720]/best",
        "quiet": True,
        "noplaylist": True,
        "socket_timeout": 20,
        "live_from_start": False,
    }

    print("Łączenie z transmisją...")

    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(
            YOUTUBE_URL,
            download=False,
        )

    if not info or not info.get("url"):
        raise RuntimeError(
            "Nie udało się uzyskać adresu strumienia."
        )

    print("Transmisja:", info.get("title", "YouTube"))

    return info["url"]


def resize_frame(frame):
    height, width = frame.shape[:2]
    new_height = round(height * FRAME_WIDTH / width)

    return cv2.resize(
        frame,
        (FRAME_WIDTH, new_height),
    )


def pixel_rect(rect, frame):
    height, width = frame.shape[:2]

    return tuple(
        round(value * dimension)
        for value, dimension in zip(
            rect,
            (width, height, width, height),
        )
    )


def inside(point, rect):
    x1, y1, x2, y2 = rect

    return (
        x1 <= point[0] <= x2
        and y1 <= point[1] <= y2
    )


# ============================================================
# EDYTOR STREF
# ============================================================

def edit_zones(frame, original):
    config = deepcopy(original)
    height, width = frame.shape[:2]

    window = "Edytor stref"
    mode = "parking"
    selected = None

    drag_start = None
    drag_end = None
    action = None

    message = "Wybierz tryb i przeciagnij mysz po obrazie."

    buttons = [
        ("Postoj [P]", "parking", 10, 170),
        ("Dodaj pomiar [M]", "measurement", 180, 390),
        ("Zaznacz [V]", "select", 400, 570),
        ("Usun [D]", "delete", 580, 740),
        ("Zapisz [ENTER]", "save", 750, 980),
    ]

    def image_point(x, y):
        return (
            max(0, min(width - 1, x)),
            max(0, min(height - 1, y - HEADER_HEIGHT)),
        )

    def mouse(event, x, y, flags, parameter):
        nonlocal drag_start
        nonlocal drag_end
        nonlocal selected
        nonlocal action

        if event == cv2.EVENT_LBUTTONDOWN:
            if y < HEADER_HEIGHT:
                for _, name, left, right in buttons:
                    if left <= x <= right and 8 <= y <= 42:
                        action = name
                        return
                return

            point = image_point(x, y)

            if mode == "select":
                selected = None

                for zone in reversed(config["measurements"]):
                    rect = pixel_rect(zone["rect"], frame)

                    if inside(point, rect):
                        selected = ("measurement", zone["id"])
                        return

                if config["parking"] is not None:
                    rect = pixel_rect(config["parking"], frame)

                    if inside(point, rect):
                        selected = ("parking", None)

                return

            drag_start = point
            drag_end = point

        elif (
            event == cv2.EVENT_MOUSEMOVE
            and drag_start is not None
        ):
            drag_end = image_point(x, y)

        elif (
            event == cv2.EVENT_LBUTTONUP
            and drag_start is not None
        ):
            drag_end = image_point(x, y)

            x1, x2 = sorted(
                (drag_start[0], drag_end[0])
            )
            y1, y2 = sorted(
                (drag_start[1], drag_end[1])
            )

            if x2 - x1 >= 16 and y2 - y1 >= 16:
                rect = [
                    x1 / width,
                    y1 / height,
                    x2 / width,
                    y2 / height,
                ]

                if mode == "parking":
                    config["parking"] = rect
                    selected = ("parking", None)

                elif mode == "measurement":
                    zone_id = uuid.uuid4().hex[:8]

                    number = 1
                    existing_names = {
                        zone["name"]
                        for zone in config["measurements"]
                    }

                    while f"Pomiar {number}" in existing_names:
                        number += 1

                    config["measurements"].append({
                        "id": zone_id,
                        "name": f"Pomiar {number}",
                        "rect": rect,
                    })

                    selected = ("measurement", zone_id)

            drag_start = None
            drag_end = None

    cv2.namedWindow(
        window,
        cv2.WINDOW_AUTOSIZE,
    )

    cv2.setMouseCallback(window, mouse)

    try:
        while True:
            canvas = np.full(
                (height + HEADER_HEIGHT, width, 3),
                25,
                dtype=np.uint8,
            )

            canvas[HEADER_HEIGHT:] = frame

            for label, name, left, right in buttons:
                color = (80, 80, 80)

                if name == mode:
                    color = (130, 90, 30)

                cv2.rectangle(
                    canvas,
                    (left, 8),
                    (right, 42),
                    color,
                    -1,
                )

                cv2.putText(
                    canvas,
                    label,
                    (left + 8, 31),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

            cv2.putText(
                canvas,
                message[:150],
                (10, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (230, 230, 230),
                1,
            )

            cv2.putText(
                canvas,
                "Zolty: postoj | Niebieski: liczenie aut | ESC: anuluj",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (230, 230, 230),
                1,
            )

            zones = []

            if config["parking"] is not None:
                zones.append((
                    ("parking", None),
                    "POSTOJ",
                    config["parking"],
                    PARKING_COLOR,
                ))

            for zone in config["measurements"]:
                zones.append((
                    ("measurement", zone["id"]),
                    zone["name"],
                    zone["rect"],
                    MEASUREMENT_COLOR,
                ))

            for identity, label, rect, color in zones:
                x1, y1, x2, y2 = pixel_rect(rect, frame)

                thickness = (
                    4 if selected == identity else 2
                )

                cv2.rectangle(
                    canvas,
                    (x1, y1 + HEADER_HEIGHT),
                    (x2, y2 + HEADER_HEIGHT),
                    color,
                    thickness,
                )

                cv2.putText(
                    canvas,
                    label,
                    (x1 + 4, y1 + HEADER_HEIGHT + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

            if (
                drag_start is not None
                and drag_end is not None
            ):
                cv2.rectangle(
                    canvas,
                    (
                        drag_start[0],
                        drag_start[1] + HEADER_HEIGHT,
                    ),
                    (
                        drag_end[0],
                        drag_end[1] + HEADER_HEIGHT,
                    ),
                    (255, 255, 255),
                    2,
                )

            cv2.imshow(window, canvas)

            key = cv2.waitKey(20) & 0xFF

            keyboard_actions = {
                ord("p"): "parking",
                ord("m"): "measurement",
                ord("v"): "select",
                ord("d"): "delete",
                13: "save",
                10: "save",
            }

            if key in keyboard_actions:
                action = keyboard_actions[key]

            if (
                key == 27
                or cv2.getWindowProperty(
                    window,
                    cv2.WND_PROP_VISIBLE,
                ) < 1
            ):
                if (
                    original.get("parking") is not None
                    and original.get("measurements")
                ):
                    return deepcopy(original)

                return None

            if action in (
                "parking",
                "measurement",
                "select",
            ):
                mode = action

                if mode == "select":
                    message = (
                        "Kliknij strefe, potem wybierz Usun."
                    )
                else:
                    message = (
                        "Przeciagnij mysz, aby narysowac strefe."
                    )

            elif action == "delete":
                if selected is not None:
                    kind, zone_id = selected

                    if kind == "parking":
                        config["parking"] = None

                    else:
                        config["measurements"] = [
                            zone
                            for zone in config["measurements"]
                            if zone["id"] != zone_id
                        ]

                    selected = None
                    message = "Usunieto zaznaczona strefe."

            elif action == "save":
                if (
                    config["parking"] is None
                    or not config["measurements"]
                ):
                    message = (
                        "Ustaw strefe postoju i minimum "
                        "jedna strefe pomiarowa."
                    )

                else:
                    write_json(CONFIG_FILE, config)
                    return config

            action = None

    finally:
        cv2.destroyWindow(window)


# ============================================================
# CZAS MATERIAŁU
# ============================================================

class VideoClock:
    """
    Pomiar czasu według materiału.

    Jeżeli znaczniki czasu są niedostępne, przyjmujemy
    czas jednej klatki na podstawie FPS źródła.

    Celowe spowolnienie wyświetlania nie wydłuża
    naliczanego czasu postoju.
    """

    def __init__(self, fps):
        self.step = 1.0 / fps
        self.previous_pts = None
        self.elapsed = 0.0
        self.started = False
        self.source = "fps"
        self.discontinuity = False

    def update(self, capture):
        pts = (
            capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
        )

        valid = math.isfinite(pts) and pts >= 0

        self.discontinuity = False
        delta = self.step
        self.source = "fps"

        if valid and self.previous_pts is not None:
            difference = pts - self.previous_pts

            if 0 < difference <= MAX_MISSING_SECONDS:
                delta = difference
                self.source = "znaczniki_materialu"

            elif (
                difference < 0
                or difference > MAX_MISSING_SECONDS
            ):
                self.discontinuity = True

        if self.started:
            self.elapsed += delta

        self.started = True
        self.previous_pts = pts if valid else None

        return self.elapsed


# ============================================================
# SESJA I ZLICZANIE POJAZDÓW
# ============================================================

class Session:
    def __init__(self, config):
        stamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        self.directory = (
            OUTPUT_DIR / f"sesja_{stamp}"
        )

        self.directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.config = deepcopy(config)

        self.started_utc = datetime.now(
            timezone.utc
        ).isoformat()

        self.counts = {
            zone["id"]: 0
            for zone in config["measurements"]
        }

        # Osobny zbiór policzonych ID dla każdej strefy.
        # ID pozostaje w zbiorze do końca sesji.
        self.counted = {
            zone["id"]: set()
            for zone in config["measurements"]
        }

        self.csv_path = (
            self.directory / "pojazdy.csv"
        )

        with self.csv_path.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as file:
            csv.writer(
                file,
                delimiter=";",
            ).writerow([
                "czas_zapisu_utc",
                "strefa_id",
                "strefa_nazwa",
                "id_trackera",
                "pierwsze_wykrycie_w_strefie_s",
                "licznik_strefy",
                "zrodlo_czasu",
            ])

        write_json(
            self.directory / "strefy.json",
            self.config,
        )

        self.summary(False)

        print("Wyniki sesji:", self.directory)

    def summary(self, finished):
        data = {
            "start_utc": self.started_utc,
            "aktualizacja_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "sesja_zakonczona": finished,
            "transmisja": YOUTUBE_URL,
            "zasada_zliczania": (
                "Pierwsze wykrycie ID w strefie; "
                "jeden raz na strefe i sesje."
            ),
            "pomiary": [
                {
                    "id": zone["id"],
                    "nazwa": zone["name"],
                    "liczba_pojazdow": self.counts[
                        zone["id"]
                    ],
                }
                for zone in self.config["measurements"]
            ],
        }

        write_json(
            self.directory / "podsumowanie.json",
            data,
        )

    def count_vehicle(
        self,
        track_id,
        point,
        timestamp,
        frame,
        clock_source,
    ):
        for zone in self.config["measurements"]:
            zone_id = zone["id"]
            rect = pixel_rect(zone["rect"], frame)

            if not inside(point, rect):
                continue

            if track_id in self.counted[zone_id]:
                continue

            self.counted[zone_id].add(track_id)
            self.counts[zone_id] += 1

            with self.csv_path.open(
                "a",
                newline="",
                encoding="utf-8",
            ) as file:
                csv.writer(
                    file,
                    delimiter=";",
                ).writerow([
                    datetime.now(
                        timezone.utc
                    ).isoformat(),
                    zone_id,
                    zone["name"],
                    track_id,
                    round(timestamp, 3),
                    self.counts[zone_id],
                    clock_source,
                ])

            self.summary(False)

            print(
                f"{zone['name']}: policzono auto "
                f"{track_id}, razem "
                f"{self.counts[zone_id]}"
            )


# ============================================================
# POSTÓJ I ZAPIS ZDJĘĆ
# ============================================================

def new_stop(position, timestamp, crop):
    return {
        "anchor": position,
        "position": position,
        "start": timestamp,
        "last": timestamp,
        "crop": crop,
        "directory": None,
        "last_write": float("-inf"),
    }


def save_stop(
    session,
    track_id,
    state,
    reason,
    clock_source,
):
    duration = max(
        0,
        state["last"] - state["start"],
    )

    if duration < MIN_STOP_SECONDS:
        return

    points = int(
        duration // SECONDS_PER_POINT
    )

    if state["directory"] is None:
        stamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        state["directory"] = (
            session.directory
            / "postoje"
            / f"{stamp}_auto_{track_id}"
        )

        state["directory"].mkdir(
            parents=True,
            exist_ok=True,
        )

    crop = state["crop"]
    padding = max(0, 480 - crop.shape[1])

    image = cv2.copyMakeBorder(
        crop,
        45,
        0,
        0,
        padding,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    caption = (
        f"ID {track_id} | "
        f"Stop: {duration:.1f}s | "
        f"Points: {points}"
    )

    cv2.putText(
        image,
        caption,
        (8, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
    )

    success, encoded = cv2.imencode(
        ".jpg",
        image,
    )

    if not success:
        raise RuntimeError(
            "Nie udało się zapisać zdjęcia."
        )

    encoded.tofile(
        str(state["directory"] / "auto.jpg")
    )

    information = {
        "id_trackera": track_id,
        "czas_postoju_s": round(duration, 2),
        "punkty": points,
        "sekundy_na_punkt": SECONDS_PER_POINT,
        "status": reason,
        "poczatek_s": round(state["start"], 3),
        "koniec_s": round(state["last"], 3),
        "zrodlo_czasu": clock_source,
        "aktualizacja_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "zdjecie": "auto.jpg",
    }

    write_json(
        state["directory"] / "informacje.json",
        information,
    )

    state["last_write"] = state["last"]


# ============================================================
# ANALIZA JEDNEJ SESJI
# ============================================================

def run_session():
    capture = None
    session = None
    clock = None

    stops = {}
    last_seen = {}

    def finish_stop(track_id, reason):
        state = stops.pop(track_id)

        save_stop(
            session,
            track_id,
            state,
            reason,
            clock.source,
        )

    try:
        capture = cv2.VideoCapture(
            get_stream_url(),
            cv2.CAP_FFMPEG,
        )

        if not capture.isOpened():
            raise RuntimeError(
                "Nie udało się otworzyć transmisji."
            )

        success, raw_frame = capture.read()

        if not success:
            raise RuntimeError(
                "Nie udało się odczytać pierwszej klatki."
            )

        frame = resize_frame(raw_frame)

        config = edit_zones(
            frame,
            load_config(),
        )

        if config is None:
            return False

        device = (
            0 if torch.cuda.is_available() else "cpu"
        )

        if device == 0:
            print(
                "Urządzenie: GPU",
                torch.cuda.get_device_name(0),
            )
        else:
            print("Urządzenie: CPU")

        # Nowa sesja = nowy tracker.
        model = YOLO(MODEL)

        source_fps = capture.get(
            cv2.CAP_PROP_FPS
        )

        if (
            not math.isfinite(source_fps)
            or not 5 <= source_fps <= 120
        ):
            source_fps = FALLBACK_FPS

        # Nie przyspieszamy źródła, jeżeli ma mniej niż 15 FPS.
        display_fps = min(
            source_fps,
            TARGET_FPS,
        )

        interval = 1.0 / display_fps

        # Zegar nadal korzysta z FPS materiału,
        # a nie ze spowolnionego wyświetlania.
        clock = VideoClock(source_fps)

        session = Session(config)

        print(f"FPS źródła: {source_fps:.1f}")
        print(
            "Maksymalne tempo odtwarzania: "
            f"{display_fps:.1f} FPS"
        )
        print(
            "Auto jest liczone przy pierwszej "
            "obserwacji w strefie."
        )

        window = (
            "YOLO - Q: koniec | E: edytor / nowa sesja"
        )

        cv2.namedWindow(
            window,
            cv2.WINDOW_NORMAL,
        )

        cv2.resizeWindow(
            window,
            1280,
            720,
        )

        next_frame_time = time.perf_counter()

        while True:
            timestamp = clock.update(capture)

            if clock.discontinuity:
                for track_id in list(stops):
                    finish_stop(
                        track_id,
                        "nieciaglosc_czasu",
                    )

                last_seen.clear()

                # Zbiorów policzonych ID nie czyścimy.
                # Nadal obowiązuje jeden zapis ID na strefę.

            parking_rect = pixel_rect(
                config["parking"],
                frame,
            )

            result = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[2],
                conf=0.25,
                imgsz=640,
                device=device,
                verbose=False,
            )[0]

            view = frame.copy()
            height, width = frame.shape[:2]

            seen = set()

            if result.boxes.id is not None:
                ids = (
                    result.boxes.id
                    .int()
                    .cpu()
                    .tolist()
                )

                boxes = (
                    result.boxes.xyxy
                    .int()
                    .cpu()
                    .tolist()
                )

                for track_id, box in zip(ids, boxes):
                    x1, y1, x2, y2 = box

                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(width, x2)
                    y2 = min(height, y2)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    seen.add(track_id)
                    last_seen[track_id] = timestamp

                    center = (
                        (x1 + x2) / 2,
                        (y1 + y2) / 2,
                    )

                    # Przybliżony punkt położenia auta na jezdni.
                    ground_point = (
                        (x1 + x2) / 2,
                        y1 + 0.9 * (y2 - y1),
                    )

                    # Liczenie przy pierwszej obecności w strefie.
                    session.count_vehicle(
                        track_id,
                        ground_point,
                        timestamp,
                        frame,
                        clock.source,
                    )

                    color = (0, 255, 0)
                    label = f"ID {track_id}"

                    # Osobna logika głównej strefy postoju.
                    if not inside(
                        ground_point,
                        parking_rect,
                    ):
                        if track_id in stops:
                            finish_stop(
                                track_id,
                                "opuscil_strefe",
                            )

                    else:
                        crop = frame[
                            y1:y2,
                            x1:x2,
                        ].copy()

                        state = stops.get(track_id)

                        if (
                            state is not None
                            and timestamp - state["last"]
                            > MAX_MISSING_SECONDS
                        ):
                            finish_stop(
                                track_id,
                                "przerwa_obserwacji",
                            )

                            state = None

                        if state is not None:
                            old_x, old_y = (
                                state["position"]
                            )

                            position = (
                                0.6 * old_x
                                + 0.4 * center[0],
                                0.6 * old_y
                                + 0.4 * center[1],
                            )

                            distance = math.dist(
                                position,
                                state["anchor"],
                            )

                            if (
                                distance
                                > MOVEMENT_TOLERANCE
                            ):
                                finish_stop(
                                    track_id,
                                    "wykryto_ruch",
                                )

                                state = None

                            else:
                                state["position"] = (
                                    position
                                )

                        if state is None:
                            state = new_stop(
                                center,
                                timestamp,
                                crop,
                            )

                            stops[track_id] = state

                        state["last"] = timestamp
                        state["crop"] = crop

                        duration = (
                            timestamp - state["start"]
                        )

                        points = int(
                            duration // SECONDS_PER_POINT
                        )

                        if (
                            duration
                            >= MIN_STOP_SECONDS
                        ):
                            color = (0, 0, 255)

                            label += (
                                f" STOP {duration:.1f}s"
                                f" PKT {points}"
                            )

                            if (
                                timestamp
                                - state["last_write"]
                                >= 1
                            ):
                                save_stop(
                                    session,
                                    track_id,
                                    state,
                                    "postoj_trwa",
                                    clock.source,
                                )

                        else:
                            color = (0, 165, 255)

                            label += (
                                " obserwacja "
                                f"{duration:.1f}s"
                            )

                    cv2.rectangle(
                        view,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2,
                    )

                    cv2.putText(
                        view,
                        label,
                        (x1, max(18, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )

                    cv2.circle(
                        view,
                        tuple(map(int, ground_point)),
                        4,
                        color,
                        -1,
                    )

            # Usuwamy stare obserwacje, ale nie historię zliczeń.
            for track_id in list(last_seen):
                if (
                    track_id not in seen
                    and timestamp - last_seen[track_id]
                    > MAX_MISSING_SECONDS
                ):
                    last_seen.pop(track_id)

                    if track_id in stops:
                        finish_stop(
                            track_id,
                            "utracono_obiekt",
                        )

            # Rysowanie stref po aktualizacji liczników.
            x1, y1, x2, y2 = parking_rect

            cv2.rectangle(
                view,
                (x1, y1),
                (x2, y2),
                PARKING_COLOR,
                2,
            )

            cv2.putText(
                view,
                "POSTOJ",
                (x1 + 4, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                PARKING_COLOR,
                2,
            )

            for zone in config["measurements"]:
                x1, y1, x2, y2 = pixel_rect(
                    zone["rect"],
                    frame,
                )

                count = session.counts[
                    zone["id"]
                ]

                cv2.rectangle(
                    view,
                    (x1, y1),
                    (x2, y2),
                    MEASUREMENT_COLOR,
                    2,
                )

                cv2.putText(
                    view,
                    f"{zone['name']}: {count}",
                    (x1 + 4, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    MEASUREMENT_COLOR,
                    2,
                )

            cv2.rectangle(
                view,
                (0, 0),
                (width, 38),
                (0, 0, 0),
                -1,
            )

            header = (
                f"Cars: {len(seen)} | "
                f"Playback max: {display_fps:.0f} FPS | "
                "E: editor/new session | Q: exit"
            )

            cv2.putText(
                view,
                header,
                (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                window,
                view,
            )

            # Ograniczenie tempa bez pomijania klatek.
            next_frame_time += interval
            now = time.perf_counter()

            # Nie przyspieszamy po wcześniejszym opóźnieniu.
            if now > next_frame_time:
                next_frame_time = now

            wait_ms = max(
                1,
                math.ceil(
                    (next_frame_time - now) * 1000
                ),
            )

            key = cv2.waitKey(wait_ms) & 0xFF

            if key == ord("e"):
                return True

            if (
                key == ord("q")
                or cv2.getWindowProperty(
                    window,
                    cv2.WND_PROP_VISIBLE,
                ) < 1
            ):
                return False

            success, raw_frame = capture.read()

            if not success:
                print("Strumień został przerwany.")
                return False

            frame = resize_frame(raw_frame)

    finally:
        try:
            if session is not None:
                for track_id in list(stops):
                    finish_stop(
                        track_id,
                        "koniec_sesji",
                    )

                session.summary(True)

        finally:
            if capture is not None:
                capture.release()

            cv2.destroyAllWindows()


# ============================================================
# START PROGRAMU
# ============================================================

def main():
    if TARGET_FPS <= 0:
        raise ValueError(
            "TARGET_FPS musi być większe od zera."
        )

    if SECONDS_PER_POINT <= 0:
        raise ValueError(
            "SECONDS_PER_POINT musi być większe od zera."
        )

    OUTPUT_DIR.mkdir(
        exist_ok=True,
    )

    while run_session():
        print(
            "Uruchamianie edytora i nowej sesji..."
        )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\nZakończono.")

    except Exception as error:
        print(f"\nBŁĄD: {error}")