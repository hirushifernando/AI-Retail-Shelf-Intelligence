from pathlib import Path
import shutil

project = Path.cwd()

raw_annotations = project / "data" / "raw" / "annotations" / "train"

train_images = project / "data" / "processed" / "yolo" / "images" / "train"
val_images = project / "data" / "processed" / "yolo" / "images" / "val"

train_labels = project / "data" / "processed" / "yolo" / "labels" / "train"
val_labels = project / "data" / "processed" / "yolo" / "labels" / "val"

train_labels.mkdir(parents=True, exist_ok=True)
val_labels.mkdir(parents=True, exist_ok=True)

xml_files = {
    xml.stem: xml
    for xml in raw_annotations.glob("*.xml")
}

print(f"XML files found: {len(xml_files)}")

train_copied = 0
train_missing = []

for image in train_images.iterdir():
    if image.is_file():
        xml = xml_files.get(image.stem)

        if xml is not None:
            shutil.copy2(xml, train_labels / xml.name)
            train_copied += 1
        else:
            train_missing.append(image.name)

val_copied = 0
val_missing = []

for image in val_images.iterdir():
    if image.is_file():
        xml = xml_files.get(image.stem)

        if xml is not None:
            shutil.copy2(xml, val_labels / xml.name)
            val_copied += 1
        else:
            val_missing.append(image.name)

print(f"Training labels copied: {train_copied}")
print(f"Validation labels copied: {val_copied}")
print(f"Training images without XML: {len(train_missing)}")
print(f"Validation images without XML: {len(val_missing)}")

if train_missing:
    print("\nMissing training XML:")
    for name in train_missing:
        print(name)

if val_missing:
    print("\nMissing validation XML:")
    for name in val_missing:
        print(name)
