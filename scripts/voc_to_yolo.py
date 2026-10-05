from pathlib import Path
import xml.etree.ElementTree as ET


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

LABELS_DIR = PROJECT_ROOT / "data" / "processed" / "yolo" / "labels"


# Class names and their YOLO class IDs
CLASS_NAMES = {
    "aqua": 0,
    "chitato": 1,
    "indomie": 2,
    "pepsodent": 3,
    "shampoo": 4,
    "tissue": 5
}


def convert_xml_to_yolo(xml_path, txt_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Get image size
    size = root.find("size")
    image_width = int(size.find("width").text)
    image_height = int(size.find("height").text)

    yolo_lines = []

    # Process every object
    for obj in root.findall("object"):
        class_name = obj.find("name").text.strip()

        if class_name not in CLASS_NAMES:
            print(f"WARNING: Unknown class '{class_name}' in {xml_path.name}")
            continue

        class_id = CLASS_NAMES[class_name]

        # Get bounding box
        bbox = obj.find("bndbox")

        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # Convert Pascal VOC coordinates to YOLO format
        x_center = ((xmin + xmax) / 2) / image_width
        y_center = ((ymin + ymax) / 2) / image_height

        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height

        yolo_lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} "
            f"{width:.6f} {height:.6f}"
        )

    # Save YOLO annotation
    txt_path.write_text("\n".join(yolo_lines), encoding="utf-8")


def convert_split(split):
    split_dir = LABELS_DIR / split

    xml_files = list(split_dir.glob("*.xml"))

    print(f"\nProcessing {split}...")
    print(f"XML files found: {len(xml_files)}")

    converted = 0

    for xml_path in xml_files:
        txt_path = split_dir / f"{xml_path.stem}.txt"

        convert_xml_to_yolo(xml_path, txt_path)

        converted += 1

    print(f"Converted: {converted}")


def main():
    print("Pascal VOC XML → YOLO TXT conversion")
    print("--------------------------------------")

    for split in ["train", "val", "test"]:
        convert_split(split)

    print("\nConversion completed.")


if __name__ == "__main__":
    main()