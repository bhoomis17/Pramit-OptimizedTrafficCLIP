import os
import csv
import random
import numpy as np
from sklearn.model_selection import train_test_split
from scapy.utils import RawPcapReader
from scapy.layers.l2 import Ether

from src.preprocessing.flow_extractor import get_flow_key
from src.preprocessing.pcap_to_image import packets_to_image
from src.preprocessing.statistics import calculate_statistics


SEED = 62
SAMPLES_PER_CLASS = 900

random.seed(SEED)
np.random.seed(SEED)

CLASSES = {
    "Skype": r".\data\Benign\Skype.pcap",
    "MySQL": r".\data\Benign\MySQL.pcap",
    "BitTorrent": r".\data\Benign\BitTorrent.pcap",
    "Facetime": r".\data\Benign\Facetime.pcap",

    "Weibo": [
        r".\data\Benign\Weibo\Weibo\Weibo-1.pcap",
        r".\data\Benign\Weibo\Weibo\Weibo-2.pcap",
        r".\data\Benign\Weibo\Weibo\Weibo-3.pcap",
        r".\data\Benign\Weibo\Weibo\Weibo-4.pcap",
    ],

    "Zeus": r".\data\Malware\Zeus.pcap",
    "Tinba": r".\data\Malware\Tinba.pcap",
    "Cridex": r".\data\Malware\Cridex\Cridex.pcap",
    "Geodo": r".\data\Malware\Geodo\Geodo.pcap",
    "Miuref": r".\data\Malware\Miuref.pcap",
}

OUTPUT_ROOT = r".\data\real_dataset"
IMAGE_ROOT = os.path.join(OUTPUT_ROOT, "images")
CSV_ROOT = os.path.join(OUTPUT_ROOT, "csv")

os.makedirs(IMAGE_ROOT, exist_ok=True)
os.makedirs(CSV_ROOT, exist_ok=True)


def get_paths(value):
    if isinstance(value, list):
        return value
    return [value]


def collect_flows(paths, needed):
    """
    Read PCAPs incrementally and stop after enough unique 5-tuple flows.
    """
    flows = {}

    for path in paths:
        if len(flows) >= needed:
            break

        print(f"  Reading: {path}")

        packet_count = 0
        reader = RawPcapReader(path)

        try:
            for raw_data, packet_metadata in reader:
                packet_count += 1

                try:
                    packet = Ether(raw_data)

                    if packet.type == 0x0800 and not packet.haslayer("IP"):
                        from scapy.layers.inet import IP
                        packet.payload = IP(bytes(packet.payload))

                    key = get_flow_key(packet)

                    if key is not None:
                        if key not in flows:
                            flows[key] = []

                        flows[key].append(packet)

                except Exception:
                    continue

                if len(flows) >= needed:
                    break

        finally:
            reader.close()

        print(f"    Packets processed: {packet_count}")
        print(f"    Flows collected: {len(flows)}")

    return list(flows.values())
def create_samples():
    rows = []

    for class_name, path_value in CLASSES.items():

        print("\n" + "=" * 60)
        print(f"CLASS: {class_name}")
        print("=" * 60)

        # Collect only the required number of flows.
        flows = collect_flows(
            get_paths(path_value),
            SAMPLES_PER_CLASS
        )

        print(f"Total flows collected: {len(flows)}")

        if len(flows) < SAMPLES_PER_CLASS:
            raise RuntimeError(
                f"{class_name} has only {len(flows)} flows, "
                f"but {SAMPLES_PER_CLASS} are required."
            )

        random.shuffle(flows)
        selected = flows[:SAMPLES_PER_CLASS]

        class_dir = os.path.join(IMAGE_ROOT, class_name)
        os.makedirs(class_dir, exist_ok=True)

        for i, flow_packets in enumerate(selected):

            image = packets_to_image(flow_packets)

            image_path = os.path.join(
                class_dir,
                f"{class_name}_{i:04d}.png"
            )

            image.save(image_path)

            stats = calculate_statistics(flow_packets)

            rows.append({
                "image_path": image_path,
                "label": class_name,
                "mean_iat": float(stats[0]),
                "jitter": float(stats[1]),
                "entropy": float(stats[2]),
            })

            if (i + 1) % 100 == 0:
                print(f"  Created {i + 1}/{SAMPLES_PER_CLASS}")

    return rows


def write_csv(filename, rows):
    path = os.path.join(CSV_ROOT, filename)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_path",
                "label",
                "mean_iat",
                "jitter",
                "entropy",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {path}")


def main():

    print("=" * 60)
    print("REAL USTC-TFC2016 DATASET PREPARATION")
    print("=" * 60)

    rows = create_samples()

    print("\nTotal samples:", len(rows))

    # Stratified 70/15/15 split
    indices = np.arange(len(rows))
    labels = np.array([r["label"] for r in rows])

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.30,
        stratify=labels,
        random_state=SEED,
    )

    temp_labels = labels[temp_idx]

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        stratify=temp_labels,
        random_state=SEED,
    )

    train_rows = [rows[i] for i in train_idx]
    val_rows = [rows[i] for i in val_idx]
    test_rows = [rows[i] for i in test_idx]

    write_csv("train.csv", train_rows)
    write_csv("val.csv", val_rows)
    write_csv("test.csv", test_rows)

    def write_stats(filename, subset):
        path = os.path.join(CSV_ROOT, filename)

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                "image_path",
                "mean_iat",
                "jitter",
                "entropy",
            ])

            for r in subset:
                writer.writerow([
                    r["image_path"],
                    r["mean_iat"],
                    r["jitter"],
                    r["entropy"],
                ])

    write_stats("train_statistics.csv", train_rows)
    write_stats("val_statistics.csv", val_rows)
    write_stats("test_statistics.csv", test_rows)

    print("\nCreating NPZ dataset...")

    from PIL import Image

    images = []
    y = []
    stats = []

    class_names = sorted(CLASSES.keys())

    class_to_idx = {
        name: i for i, name in enumerate(class_names)
    }

    for r in rows:
        img = Image.open(r["image_path"]).convert("L")
        arr = np.array(img, dtype=np.uint8)

        images.append(arr[np.newaxis, :, :])

        y.append(class_to_idx[r["label"]])

        stats.append([
            r["mean_iat"],
            r["jitter"],
            r["entropy"],
        ])

    np.savez_compressed(
        os.path.join(OUTPUT_ROOT, "ustc_10class.npz"),
        x=np.array(images, dtype=np.uint8),
        y=np.array(y, dtype=np.int64),
        labels=np.array(class_names),
        m=np.array(stats, dtype=np.float32),
    )

    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)
    print("Train:", len(train_rows))
    print("Validation:", len(val_rows))
    print("Test:", len(test_rows))
    print(
        "NPZ:",
        os.path.join(OUTPUT_ROOT, "ustc_10class.npz")
    )


if __name__ == "__main__":
    main()