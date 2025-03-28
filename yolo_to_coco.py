import os
import json

# Directories
train_images_dir = r'D:\drone detection part3\yolov7\drone_dataset\train\images'
train_labels_dir = r'D:\drone detection part3\yolov7\drone_dataset\train\labels'
test_images_dir = r'D:\drone detection part3\yolov7\drone_dataset\valid\images'
test_labels_dir = r'D:\drone detection part3\yolov7\drone_dataset\valid\labels'
output_json = r'D:\drone detection part3\yolov7\annotations_coco.json'


# COCO Format template
coco_format = {
    'images': [],
    'annotations': [],
    'categories': []
}

# Add categories (modify as needed)
categories = [{'id': 1, 'name': 'drone', 'supercategory': 'object'}]
coco_format['categories'].extend(categories)

annotation_id = 1

# Function to convert YOLO to COCO format
def yolo_to_coco(image_id, label_path, image_width, image_height):
    with open(label_path, 'r') as f:
        lines = f.readlines()
        annotations = []
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
            x = (x_center - width / 2) * image_width
            y = (y_center - height / 2) * image_height
            width *= image_width
            height *= image_height
            annotation = {
                'id': annotation_id,
                'image_id': image_id,
                'category_id': class_id + 1,  # COCO IDs start from 1
                'bbox': [x, y, width, height],
                'area': width * height,
                'iscrowd': 0
            }
            annotations.append(annotation)
        return annotations

# Process images and labels
def process_dataset(image_dir, label_dir):
    global annotation_id
    for idx, image_name in enumerate(os.listdir(image_dir)):
        if not image_name.endswith(('.jpg', '.png', '.jpeg')):
            continue
        image_id = idx + 1
        image_path = os.path.join(image_dir, image_name)
        label_path = os.path.join(label_dir, os.path.splitext(image_name)[0] + '.txt')

        # Image info
        image = {
            'id': image_id,
            'file_name': image_name,
            'width': 640,  # Update with actual image size if needed
            'height': 640
        }
        coco_format['images'].append(image)

        if os.path.exists(label_path):
            annotations = yolo_to_coco(image_id, label_path, image['width'], image['height'])
            coco_format['annotations'].extend(annotations)
            annotation_id += len(annotations)

# Process train and test datasets
process_dataset(train_images_dir, train_labels_dir)
process_dataset(test_images_dir, test_labels_dir)

# Save to JSON file
with open(output_json, 'w') as outfile:
    json.dump(coco_format, outfile, indent=4)

print(f'Conversion complete! COCO format JSON saved as {output_json}')
