import json
import os


def remove_images_with_category(coco_file, category_id_to_remove, output_file):
    # Load the COCO JSON file
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)

    # Find image IDs that contain the specified category_id
    image_ids_to_remove = {ann['image_id'] for ann in coco_data['annotations'] if ann['category_id'] == category_id_to_remove}

    # Filter images
    coco_data['images'] = [img for img in coco_data['images'] if img['id'] not in image_ids_to_remove]

    # Filter annotations
    coco_data['annotations'] = [ann for ann in coco_data['annotations'] if ann['image_id'] not in image_ids_to_remove]

    # Update categories - remove categories that are no longer referenced in annotations
    coco_data['categories'] = [category for category in coco_data['categories'] 
                               if any(ann['category_id'] == category['id'] for ann in coco_data['annotations'])]

    folder_path = os.path.dirname(coco_file)

    # Save the modified JSON
    with open(os.path.join(folder_path, output_file), 'w') as f:
        json.dump(coco_data, f, indent=4)

    print(f"Removed {len(image_ids_to_remove)} images containing category_id {category_id_to_remove}")
    print(f"Saved modified annotations to {os.path.join(folder_path, output_file)}")


def print_categories_with_names(coco_file):
    # Load the COCO JSON file
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)

    # Create a dictionary mapping category_id to category name
    category_mapping = {cat['id']: cat['name'] for cat in coco_data['categories']}

    # Print category ID and name
    print("Category ID - Name Mapping:")
    for cat_id, cat_name in category_mapping.items():
        print(f"{cat_id}: {cat_name}")

def modify_category_name(coco_file, category_id_to_modify, new_name, output_file):
    # Load the COCO JSON file
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)

    # Find the category and update its name
    for category in coco_data['categories']:
        if category['id'] == category_id_to_modify:
            category['name'] = new_name
            print(f"Category ID {category_id_to_modify} name updated to '{new_name}'")
            break
    else:
        print(f"Category ID {category_id_to_modify} not found.")
    
    folder_path = os.path.dirname(coco_file)
    # Save the modified JSON
    with open(os.path.join(folder_path,output_file), 'w') as f:
        json.dump(coco_data, f, indent=4)

    print(f"Saved modified annotations to {output_file}")


# # # Example usage
# print_categories_with_names("/home/m32patel/projects/rrg-dclausi/wildlife/datasets/animal_patches/birds_Izembek_Lagoon_Waterfowl/train/train.json")

# # # Example usage
# modify_category_name("/home/m32patel/projects/rrg-dclausi/wildlife/datasets/animal_patches/birds_Izembek_Lagoon_Waterfowl/train/train.json", category_id_to_modify=6, new_name="Brant goose", output_file="modified_coco.json")

# modify_category_name("/home/m32patel/projects/rrg-dclausi/wildlife/datasets/animal_patches/birds_Izembek_Lagoon_Waterfowl/train/modified_coco.json", category_id_to_modify=3, new_name="Canada goose", output_file="modified_coco.json")

# modify_category_name("/home/m32patel/projects/rrg-dclausi/wildlife/datasets/animal_patches/birds_Izembek_Lagoon_Waterfowl/train/modified_coco.json", category_id_to_modify=4, new_name="Emperor goose", output_file="modified_coco.json")



# # # # Example usage
# print_categories_with_names("/home/m32patel/projects/rrg-dclausi/wildlife/datasets/animal_patches/birds_Izembek_Lagoon_Waterfowl/train/modified_coco.json")

# # # # Example usage
# remove_images_with_category("/home/m32patel/projects/rrg-dclausi/wildlife/datasets/animal_patches/birds_Izembek_Lagoon_Waterfowl/train/modified_coco.json", category_id_to_remove=2, output_file="train_no_gull.json")


# # # Example usage
# print_categories_with_names("/home/m32patel/projects/rrg-dclausi/wildlife/datasets/animal_patches/birds_Izembek_Lagoon_Waterfowl/train/train_no_gull.json")




# # Example usage
print_categories_with_names("/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_Izembek_Lagoon_Waterfowl/test.json")

# # Example usage
modify_category_name("/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_Izembek_Lagoon_Waterfowl/test.json", category_id_to_modify=0, new_name="Brant goose", output_file="modified_coco.json")

modify_category_name("/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_Izembek_Lagoon_Waterfowl/modified_coco.json", category_id_to_modify=3, new_name="Canada goose", output_file="modified_coco.json")

modify_category_name("/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_Izembek_Lagoon_Waterfowl/modified_coco.json", category_id_to_modify=4, new_name="Emperor goose", output_file="modified_coco.json")


# # # # Example usage
print_categories_with_names("/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_Izembek_Lagoon_Waterfowl/modified_coco.json")

# # # # Example usage
remove_images_with_category("/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_Izembek_Lagoon_Waterfowl/modified_coco.json", category_id_to_remove=2, output_file="test_no_gull.json")

# # # Example usage
print_categories_with_names("/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_Izembek_Lagoon_Waterfowl/test_no_gull.json")