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
  </PolygonLabels>

  <TextArea name="customLabel" toName="image" perRegion="true"
            editable="true" required="false" maxSubmissions="1"
            placeholder="Enter custom label" rows="1"/>

  <TextArea name="totalWeight" toName="image" editable="true" required="true" maxSubmissions="1"
            placeholder="Enter total plate weight (e.g., 100g)" rows="1"/>
</View>
```

---

## 📝 Semantic Segmentation Labels

These are the predefined food categories for polygon annotation:

- 🫑 **Gemista**
- 🥬 **Green Beans**
- 🍔 **Burgers**
- 🍗 **Chicken**
- 🍲 **Giouvetsi**
- 🧀 **Feta**
- 🥒 **Cucumber**
- 🍝 **Pasta**
- 🍖 **Minced Meat**
- 🧈 **Cheese**
- 🍚 **Rice**
- 🥦 **Okra**
- 🥪 **Toast Cheese**
- 🥚 **Eggs**
- 🍛 **Lentils**
- 🥗 **Salad Leaves**
- 🍄 **Mushrooms**
- ❓ **Other** (with optional custom label)

---


For the actual images contact me


