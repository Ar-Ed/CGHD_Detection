import os 
import requests
import tqdm
from zipfile import ZipFile
import json
import glob
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET


##### Get Dataset #####


def unzip(file_path, remove_zip=True):
    with ZipFile(file_path, 'r') as zip:
        zip.extractall(".")
    if remove_zip:
        os.remove(file_path)

def install_dataset(zip_name, url):
    response = requests.get(url, stream=True)

    total_size_in_bytes= int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm.tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(zip_name, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()


##### TO YOLO Format #####


def extract_info_from_xml(xml_file):
    root = ET.parse(xml_file).getroot()
    
    # Initialise the info dict 
    info_dict = {}
    info_dict['bboxes'] = []

    # Parse the XML Tree
    for elem in root:
        # Get the file name 
        if elem.tag == "filename":
            info_dict['filename'] = elem.text
            
        # Get the image size
        elif elem.tag == "size":
            image_size = []
            for subelem in elem:
                image_size.append(int(subelem.text))
            
            info_dict['image_size'] = tuple(image_size)
        
        # Get details of the bounding box 
        elif elem.tag == "object":
            bbox = {}
            for subelem in elem:
                if subelem.tag == "name":
                    bbox["class"] = subelem.text
                    
                elif subelem.tag == "bndbox":
                    for subsubelem in subelem:
                        bbox[subsubelem.tag] = int(subsubelem.text)            
            info_dict['bboxes'].append(bbox)
    
    return info_dict

def convert_to_yolov5(info_dict):
    
    with open(os.path.join("content", "MyDrive", "class_name_to_id_mapping.json"), "r") as f:
        class_name_to_id_mapping = json.load(f)

    print_buffer = []
    
    # For each bounding box
    for b in info_dict["bboxes"]:
        try:
            class_id = class_name_to_id_mapping[b["class"]]
        except KeyError:
            print("Invalid Class. Must be one from ", class_name_to_id_mapping.keys())
        
        # Transform the bbox co-ordinates as per the format required by YOLO v5
        b_center_x = (b["xmin"] + b["xmax"]) / 2 
        b_center_y = (b["ymin"] + b["ymax"]) / 2
        b_width    = (b["xmax"] - b["xmin"])
        b_height   = (b["ymax"] - b["ymin"])
        
        # Normalise the co-ordinates by the dimensions of the image
        image_w, image_h, image_c = info_dict["image_size"]  
        b_center_x /= image_w 
        b_center_y /= image_h 
        b_width    /= image_w 
        b_height   /= image_h 
        
        #Write the bbox details to the file 
        print_buffer.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, b_center_x, b_center_y, b_width, b_height))
        
    # Name of the file which we have to save 
    save_file_name = os.path.join("CGHD-1152-YOLO", "labels", info_dict["filename"].replace("jpg", "txt"))
    
    # Save the annotation to disk
    print("\n".join(print_buffer), file= open(save_file_name, "w"))

def enumerate_classes():
    set_ = set()
    for image_file in glob.glob(os.path.join("CGHD-1152", "*.xml")): 
        for object in extract_info_from_xml(image_file)["bboxes"]:
            set_.add(object["class"])
    class_name_to_id_mapping = {key: value for value, key in enumerate(set_, )}

    # Move this file to drive manually if you need to
    with open("class_name_to_id_mapping.json", "w") as f:
        json.dump(class_name_to_id_mapping, f)
        print("\nDumped to class_name_to_id_mapping.json\nMove This file to drive for repeated usage\n")

    return class_name_to_id_mapping

def load_classes_from_drive():
    with open(os.path.join("content", "MyDrive", "class_name_to_id_mapping.json"), "r") as f:
        class_name_to_id_mapping = json.load(f)
        class_names = list(class_name_to_id_mapping)
    return class_name_to_id_mapping, class_names


def generate_yolo_yaml_file():
    with open("CGHD-1152.yaml", "w") as f:
        f.write("""# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
        # path: ./data/CGHD-1152-YOLO  # dataset root dir
        train: ../CGHD-1152-YOLO/images/train  # train images (relative to 'path') 
        val: ../CGHD-1152-YOLO/images/val  # val images (relative to 'path')
        test: ../CGHD-1152-YOLO/images/test  # val images (relative to 'path')

        # Classes (45 CGHD classes)
        nc: 45

        names:
        ['crossover', 'microphone', 'transistor.photo', 'triac', 'capacitor.polarized', 'operational_amplifier', 'lamp', 'junction', 'voltage.dc_ac', 'diode.light_emitting', 'integrated_circuit', 'probe.current', 'optocoupler', 'terminal', 'switch', 'or', 'relay', 'nor', 'voltage.dc_regulator', 'resistor.adjustable', 'transistor', 'socket', 'diode', 'resistor.photo', 'schmitt_trigger', 'varistor', 'resistor', 'integrated_cricuit.ne555', 'antenna', 'thyristor', 'diac', 'and', 'motor', 'nand', 'inductor', 'vss', 'fuse', 'not', 'xor', 'voltage.dc', 'transformer', 'gnd', 'text', 'capacitor.unpolarized', 'speaker']"""
        )

def generate_annotations_yolo_format():
    annotations = [os.path.join("CGHD-1152", x) for x in os.listdir(os.path.join("CGHD-1152")) if x[-3:] == "xml"]
    annotations.sort()

    # Convert and save the annotations as .txt files
    for ann in tqdm.tqdm(annotations):
        info_dict = extract_info_from_xml(ann)
        convert_to_yolov5(info_dict)

def move_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.move(f, destination_folder)
        except:
            print(f)
            assert False

def copy_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.copy(f, destination_folder)
        except:
            print(f)
            assert False

def populate_yolo_folders():
    # Read images and annotations
    train_images = [image_path for i in range(1, 126) for image_path in glob.glob(os.path.join("CGHD-1152", f"C{i}_*.jpg"))]
    val_images = [image_path for i in range(126, 133) for image_path in glob.glob(os.path.join("CGHD-1152", f"C{i}_*.jpg"))]
    test_images = [image_path for i in range(133, 145) for image_path in glob.glob(os.path.join("CGHD-1152", f"C{i}_*.jpg"))]

    train_labels = [image_path for i in range(1, 126) for image_path in glob.glob(os.path.join("CGHD-1152-YOLO", "labels", f"C{i}_*.txt"))]
    val_labels = [image_path for i in range(126, 133) for image_path in glob.glob(os.path.join("CGHD-1152-YOLO", "labels", f"C{i}_*.txt"))]
    test_labels = [image_path for i in range(133, 145) for image_path in glob.glob(os.path.join("CGHD-1152-YOLO", "labels", f"C{i}_*.txt"))]


    # Move the splits into their folders
    copy_files_to_folder(train_images, os.path.join("CGHD-1152-YOLO", "images", "train"))
    copy_files_to_folder(val_images, os.path.join("CGHD-1152-YOLO", "images", "val"))
    copy_files_to_folder(test_images, os.path.join("CGHD-1152-YOLO", "images", "test"))
    move_files_to_folder(train_labels, os.path.join("CGHD-1152-YOLO", "labels", "train"))
    move_files_to_folder(test_labels, os.path.join("CGHD-1152-YOLO", "labels", "test"))
    move_files_to_folder(val_labels, os.path.join("CGHD-1152-YOLO", "labels", "val"))


##### TO COCO FORMAT #####


def setup_for_voc2coco(class_name_to_id_mapping):
    os.mkdir("Annotations")
    os.mkdir("dataset_ids")

    with open("labels.txt", "w") as f:
        f.write("\n".join(list(class_name_to_id_mapping)))

    with open(os.path.join("dataset_ids", "train.txt"), "w") as f:
        f.write("\n".join([Path(image_path).stem + ".xml" for i in range(1, 126) for image_path in glob.glob(os.path.join("CGHD-1152", f"C{i}_*.xml"))]))

    with open(os.path.join("dataset_ids", "val.txt"), "w") as f:
        f.write("\n".join([Path(image_path).stem + ".xml" for i in range(126, 133) for image_path in glob.glob(os.path.join("CGHD-1152", f"C{i}_*.xml"))]))

    with open(os.path.join("dataset_ids", "test.txt"), "w") as f:
        f.write("\n".join([Path(image_path).stem + ".xml" for i in range(133, 145) for image_path in glob.glob(os.path.join("CGHD-1152", f"C{i}_*.xml"))]))


    annotations = [image_path for image_path in glob.glob(os.path.join("CGHD-1152", f"*.xml"))]
    copy_files_to_folder(annotations, os.path.join("Annotations"))

def cleanup_voc2coco():
    shutil.rmtree("Annotations")
    shutil.rmtree("dataset_ids")
    os.remove("labels.txt")