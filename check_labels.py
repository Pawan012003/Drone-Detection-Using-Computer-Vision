import os

def check_labels(label_dir):
    for root, _, files in os.walk(label_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            print(f"Error in {file_path}: {line.strip()}")
                        else:
                            class_id, x_center, y_center, width, height = parts
                            try:
                                class_id = int(class_id)
                                x_center = float(x_center)
                                y_center = float(y_center)
                                width = float(width)
                                height = float(height)

                                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                                    print(f"Invalid values in {file_path}: {line.strip()}")
                            except ValueError:
                                print(f"Non-numeric value in {file_path}: {line.strip()}")

print("Checking label files...")
check_labels("D:/drone detection part3/yolov7/drone_dataset/train/labels/")
check_labels("D:/drone detection part3/yolov7/drone_dataset/val/labels/")
print("Label check completed!")

