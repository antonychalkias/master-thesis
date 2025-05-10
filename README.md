# 🍽️ Food Image Recognition, Weight Estimation & Nutrition Tool

This project provides a complete workflow for labeling, segmenting, and preparing datasets for machine learning models that perform **food image recognition**, **weight estimation**, and **nutritional analysis**.

---

## 📚 Table of Contents

- [🔧 Setup](#-setup)
- [🚀 Steps to Use](#-steps-to-use)
  - [1. Install Label Studio](#1-install-label-studio)
  - [2. Start Label Studio](#2-start-label-studio)
  - [3. Generate Labeling JSON](#3-generate-labeling-json)
  - [4. Prepare Dataset for Labeling](#4-prepare-dataset-for-labeling)
  - [5. Serve Images](#5-serve-images)
  - [6. Import JSON to Label Studio](#6-import-json-to-label-studio)
  - [7. Label Studio Template Format](#7-label-studio-template-format)
- [📝 Semantic Segmentation Labels](#-semantic-segmentation-labels)
- [🔄 Dataset Processing](#-dataset-processing)
  - [1. Combining CSV Datasets](#1-combining-csv-datasets)
  - [2. Reordering and Renaming Images](#2-reordering-and-renaming-images)

---

## 🔧 Setup

To get started, make sure you have Python installed and then install [Label Studio](https://labelstud.io/):

```bash
pip install label-studio
```

---

## 🚀 Steps to Use

### 1. Install Label Studio

```bash
pip install label-studio
```

### 2. Start Label Studio

```bash
label-studio start
```

---

### 3. Generate Labeling JSON

To overcome some Label Studio limitations, we've provided a custom script to generate the labeling JSON:

```bash
cd python_scripts
python3 generate_images_to_json.py
```

---

### 4. Prepare Dataset for Labeling

After generating the JSON, run Label Studio again:

```bash
label-studio start
```

---

### 5. Serve Images

Run the `server.py` script to serve your images with CORS enabled:

```bash
python3 server.py
```

---

### 6. Import JSON to Label Studio

In the Label Studio UI:
- Create or open a project.
- Go to **Import** and load the generated `.json` file from the previous steps.

---

### 7. Label Studio Template Format

For **semantic segmentation** with polygon labels, use the following configuration:

```xml
<View>
  <Image name="image" value="$image" zoom="true"/>

  <PolygonLabels name="label" toName="image" strokeWidth="3">
    <Label value="Gemista" background="red"/>
    <Label value="Green Beans" background="green"/>
    <Label value="Burgers" background="brown"/>
    <Label value="Chicken" background="orange"/>
    <Label value="Giouvetsi" background="purple"/>
    <Label value="Feta" background="lightblue"/>
    <Label value="Cucumber" background="darkgreen"/>
    <Label value="Pasta" background="yellow"/>
    <Label value="Minced Meat" background="maroon"/>
    <Label value="Cheese" background="gold"/>
    <Label value="Rice" background="pink"/>
    <Label value="Okra" background="olive"/>
    <Label value="Toast Cheese" background="beige"/>
    <Label value="Eggs" background="lightyellow"/>
    <Label value="Lentils" background="saddlebrown"/>
    <Label value="Salad Leaves" background="lightgreen"/>
    <Label value="Mushrooms" background="#A0522D"/>
    <Label value="Other" background="gray"/>
    <Label value="Bread" background="#FFA39E"/>
    <Label value="Potatos" background="#FFA46E"/>
    <Label value="Carrots" background="orange"/> 
    <Label value="Onions" background="#B7EB8F"/>
    <Label value="Peas" background="#B7E19F"/>
    <Label value="Snails" background="#A0500D"/>
    <Label value="Lamb" background="#AA00DD"/>
    <Label value="Broccoli" background="#90EE90"/>
    <Label value="Corn" background="yellow"/>
    
  </PolygonLabels>

  <TextArea name="other_description" toName="image"
            perRegion="true"
            visibleWhen="choice=Other"
            required="false"
            placeholder="Please describe the 'Other' item"
            rows="3"/>

  <TextArea name="totalWeight" toName="image"
            editable="true"
            required="true"
            maxSubmissions="1"
            placeholder="Enter total plate weight (e.g., 100g)"
            rows="1"/>
</View>
```

---

## 📝 Semantic Segmentation Labels

These are the predefined food categories for polygon annotation:

  - Gemista 🫑
  - Green Beans 🫘
  - Burgers 🍔
  - Chicken 🍗
  - Giouvetsi 🍲
  - Feta 🧀
  - Cucumber 🥒
  - Pasta 🍝
  - Minced Meat 🥩
  - Cheese 🧀
  - Rice 🍚
  - Okra 🌿
  - Toast Cheese 🥪
  - Eggs 🥚
  - Lentils 🍛
  - Salad Leaves 🥗
  - Mushrooms 🍄
  - Other ❓
  - Bread 🍞
  - Potatos 🥔
  - Carrots 🥕
  - Onions 🧅
  - Peas 🌱
  - Snails 🐌
  - Lamb 🐑
  - Broccoli 🥦
  - Corn 🌽


---

## 🔄 Dataset Processing

After labeling your food images, use these scripts to process and prepare your datasets for training machine learning models.

### 1. Combining CSV Datasets

The `combine_csv_datasets.py` script merges data from multiple CSV files into a single comprehensive dataset:

```bash
cd python_scripts/after_labeling_scripts
python3 combine_csv_datasets.py
```

This script:
- Combines data from `labels_for_ordered_dataset.csv` and `labels_for_dataset_foods.csv`
- Extracts food labels from polygon annotations in JSON format
- Processes image names and URLs to maintain consistency
- Creates a unified dataset with fields for image name, labels, weight, volume, energy, etc.
- Outputs a combined CSV file at `csvfiles/combined_dataset_labels.csv`

### 2. Reordering and Renaming Images

The `reorder_images_update_csv.py` script renames all images sequentially and updates the CSV references:

```bash
cd python_scripts
python3 reorder_images_update_csv.py
```

This script:
- Combines images from `dataset_foods` and `ordered_dataset` folders
- Renames all images sequentially (1.jpg, 2.jpg, etc.)
- Creates a new directory `ordered_dataset_foods_ready` with the renamed images
- Updates the CSV file to reference the new image names
- Preserves original image names in the CSV for reference
- Creates a new CSV file at `csvfiles/combined_dataset_labels_ready.csv`

---

## 📸 Image Data Access

For the dataset of annotated images contact me.

