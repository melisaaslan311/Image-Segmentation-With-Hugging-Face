# 🎨 Image Segmentation & IoU Calculation 📊

This project focuses on image segmentation and calculates the Intersection over Union (IoU) scores. 🏆
Using a pre-trained Segformer model, it analyzes clothing images and compares the predicted segmentation with the ground truth masks. 👕👗

🚀 Project Overview
📂 Dataset: Image data is obtained from the train.csv file, which includes image IDs and encoded pixel masks.
🏷️ Labels: Category and attribute details are extracted from label_descriptions.json.
🧠 Model: Utilizes a Segformer-based pre-trained image segmentation model.
📊 IoU Calculation: Compares ground truth and predicted masks to compute IoU scores.
📝 Outputs: IoU results are saved in an Excel file (IoU_degerleri.xlsx).

📌 Usage
1️⃣ Prepare the dataset: Ensure that the train.csv and label_descriptions.json files are in the project directory, along with the train/ folder containing images.
2️⃣ Run IoU calculation for a sample image:
""image_id = train_df.iloc[0]["ImageId"]
iou, pred_seg, true_mask, image = calculate_iou_for_image(f'train/{image_id}.jpg', train_df, image_id)
print(f"IoU for image {image_id}: {iou}")""
3️⃣ Compute IoU for multiple images: The script iterates through images, calculates IoU scores, and stores them in a dictionary.
4️⃣ Export Results: IoU scores and statistics (mean, standard deviation) are saved in an Excel file (IoU_degerleri.xlsx).

📊 Sample Results
Example IoU scores after computation:
![image](https://github.com/user-attachments/assets/7c29031f-8a21-4fb1-96fc-eacca2b75130)
![image](https://github.com/user-attachments/assets/c560de26-856b-4954-983b-4fe279871915)



