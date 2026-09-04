"""
disease_info.py
=====================================================
Human-readable descriptions for the 38 PlantVillage disease classes.
The AI model returns a raw class label like "Tomato___Early_blight";
this file maps that label to farmer-friendly crop name, disease name,
description, symptoms, treatment, and prevention text shown in the UI.

If the model returns a class not listed here, a generic fallback entry
is used (see get_disease_info()) so the app never crashes.
"""

DISEASE_INFO = {
    "Apple___Apple_scab": {
        "crop": "Apple", "disease": "Apple Scab",
        "description": "A fungal disease causing dark, scabby lesions on leaves and fruit.",
        "symptoms": "Olive-green to black velvety spots on leaves; dark scabby lesions on fruit surface.",
        "treatment": "Remove and destroy fallen infected leaves; consider a suitable fungicide during early season.",
        "prevention": "Choose scab-resistant varieties, prune for airflow, avoid overhead irrigation.",
    },
    "Apple___Black_rot": {
        "crop": "Apple", "disease": "Black Rot",
        "description": "A fungal disease affecting leaves, fruit, and bark.",
        "symptoms": "Purple-bordered leaf spots, rotting fruit with concentric rings, cankers on branches.",
        "treatment": "Prune out cankers and infected wood; remove mummified fruit; consult local advisory for fungicide options.",
        "prevention": "Sanitize pruning tools, remove infected plant debris, maintain tree vigor.",
    },
    "Apple___Cedar_apple_rust": {
        "crop": "Apple", "disease": "Cedar Apple Rust",
        "description": "A fungal disease requiring both apple and cedar/juniper hosts to complete its lifecycle.",
        "symptoms": "Bright orange-yellow spots on leaves, sometimes with small black dots.",
        "treatment": "Remove nearby cedar/juniper hosts if practical; consult local advisory on fungicide timing.",
        "prevention": "Plant rust-resistant apple varieties; increase distance from juniper/cedar trees.",
    },
    "Apple___healthy": {
        "crop": "Apple", "disease": "Healthy",
        "description": "No significant disease symptoms detected.",
        "symptoms": "Leaf appears structurally normal with typical green coloration.",
        "treatment": "No treatment required. Continue regular monitoring.",
        "prevention": "Maintain balanced fertilization, proper pruning, and routine pest monitoring.",
    },
    "Blueberry___healthy": {
        "crop": "Blueberry", "disease": "Healthy",
        "description": "No significant disease symptoms detected.",
        "symptoms": "Leaf appears structurally normal.",
        "treatment": "No treatment required. Continue regular monitoring.",
        "prevention": "Maintain proper soil acidity and consistent watering.",
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "crop": "Cherry", "disease": "Powdery Mildew",
        "description": "A fungal disease producing a white powdery coating on leaves.",
        "symptoms": "White powdery patches on leaf surfaces, curling or distorted young leaves.",
        "treatment": "Improve air circulation via pruning; consult local advisory for approved fungicides.",
        "prevention": "Avoid excess nitrogen fertilization, ensure adequate plant spacing.",
    },
    "Cherry_(including_sour)___healthy": {
        "crop": "Cherry", "disease": "Healthy",
        "description": "No significant disease symptoms detected.",
        "symptoms": "Leaf appears structurally normal.",
        "treatment": "No treatment required. Continue regular monitoring.",
        "prevention": "Maintain balanced fertilization and routine pest monitoring.",
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "crop": "Maize", "disease": "Gray Leaf Spot",
        "description": "A fungal disease causing rectangular lesions between leaf veins.",
        "symptoms": "Tan to gray rectangular lesions running parallel to leaf veins.",
        "treatment": "Rotate crops; consult local advisory on fungicide options for severe cases.",
        "prevention": "Use resistant hybrids, rotate away from corn for a season, manage crop residue.",
    },
    "Corn_(maize)___Common_rust_": {
        "crop": "Maize", "disease": "Common Rust",
        "description": "A fungal disease producing rust-colored pustules on leaves.",
        "symptoms": "Small, cinnamon-brown, raised pustules scattered on both leaf surfaces.",
        "treatment": "Monitor spread; consult local advisory for fungicide options in severe infestations.",
        "prevention": "Plant resistant hybrids, avoid dense planting, rotate crops seasonally.",
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "crop": "Maize", "disease": "Northern Leaf Blight",
        "description": "A fungal disease producing long cigar-shaped lesions.",
        "symptoms": "Long, elliptical, gray-green to tan lesions on lower leaves first.",
        "treatment": "Remove severely affected leaves; consult local advisory for management options.",
        "prevention": "Use resistant hybrids, rotate crops, manage residue after harvest.",
    },
    "Corn_(maize)___healthy": {
        "crop": "Maize", "disease": "Healthy",
        "description": "No significant disease symptoms detected.",
        "symptoms": "Leaf appears structurally normal.",
        "treatment": "No treatment required. Continue regular monitoring.",
        "prevention": "Maintain balanced fertilization and proper irrigation schedule.",
    },
    "Grape___Black_rot": {
        "crop": "Grape", "disease": "Black Rot",
        "description": "A fungal disease affecting leaves, shoots, and fruit.",
        "symptoms": "Circular brown leaf spots with dark borders; shriveled, mummified fruit.",
        "treatment": "Remove mummified fruit and infected debris; consult local advisory for fungicide timing.",
        "prevention": "Prune for airflow, remove infected canes, practice good sanitation.",
    },
    "Grape___Esca_(Black_Measles)": {
        "crop": "Grape", "disease": "Esca (Black Measles)",
        "description": "A complex fungal trunk disease affecting older grapevines.",
        "symptoms": "Tiger-stripe leaf discoloration, dark spotting on berries, sudden vine collapse in severe cases.",
        "treatment": "Remove and destroy severely affected wood; consult a viticulture specialist for management.",
        "prevention": "Avoid pruning wounds during wet weather, protect pruning cuts.",
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "crop": "Grape", "disease": "Leaf Blight",
        "description": "A fungal disease producing angular leaf spots.",
        "symptoms": "Brown angular spots on leaves, sometimes with yellow halos.",
        "treatment": "Remove affected leaves; consult local advisory for fungicide recommendations.",
        "prevention": "Ensure good canopy airflow, avoid overhead irrigation.",
    },
    "Grape___healthy": {
        "crop": "Grape", "disease": "Healthy",
        "description": "No significant disease symptoms detected.",
        "symptoms": "Leaf appears structurally normal.",
        "treatment": "No treatment required. Continue regular monitoring.",
        "prevention": "Maintain balanced fertilization and routine vineyard sanitation.",
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "crop": "Orange", "disease": "Citrus Greening (HLB)",
        "description": "A serious bacterial disease spread by psyllid insects, with no known cure.",
        "symptoms": "Blotchy yellow mottling on leaves, lopsided bitter fruit, twig dieback.",
        "treatment": "Remove and destroy infected trees to prevent spread; consult a citrus specialist/agricultural officer immediately.",
        "prevention": "Control psyllid insect vectors, use certified disease-free planting material.",
    },
    "Peach___Bacterial_spot": {
        "crop": "Peach", "disease": "Bacterial Spot",
        "description": "A bacterial disease affecting leaves, fruit, and twigs.",
        "symptoms": "Small dark angular spots on leaves that may fall out leaving a 'shot-hole' look; sunken fruit lesions.",
        "treatment": "Remove severely infected material; consult local advisory for bactericide options.",
        "prevention": "Use resistant varieties, avoid overhead irrigation, prune for airflow.",
    },
    "Peach___healthy": {
        "crop": "Peach", "disease": "Healthy",
        "description": "No significant disease symptoms detected.",
        "symptoms": "Leaf appears structurally normal.",
        "treatment": "No treatment required. Continue regular monitoring.",
        "prevention": "Maintain balanced fertilization and routine pest monitoring.",
    },
    "Pepper,_bell___Bacterial_spot": {
        "crop": "Bell Pepper", "disease": "Bacterial Spot",
        "description": "A bacterial disease causing leaf and fruit lesions.",
        "symptoms": "Small water-soaked spots turning dark brown, leaf yellowing, raised scabby fruit spots.",
        "treatment": "Remove infected plants/debris; consult local advisory for approved bactericide.",
        "prevention": "Use certified disease-free seed, avoid overhead watering, rotate crops.",
    },
    "Pepper,_bell___healthy": {
        "crop": "Bell Pepper", "disease": "Healthy",
        "description": "No significant disease symptoms detected.",
        "symptoms": "Leaf appears structurally normal.",
        "treatment": "No treatment required. Continue regular monitoring.",
        "prevention": "Maintain balanced fertilization and proper spacing.",
    },
    "Potato___Early_blight": {
        "crop": "Potato", "disease": "Early Blight",
        "description": "A common fungal disease causing dark concentric spots on lower leaves.",
        "symptoms": "Brown/black concentric ring spots on older leaves, yellowing around spots.",
        "treatment": "Remove and destroy affected leaves; consult local advisory for fungicide options.",
        "prevention": "Crop rotation, avoid overhead watering, use disease-resistant varieties.",
    },
    "Potato___Late_blight": {
        "crop": "Potato", "disease": "Late Blight",
        "description": "A serious disease that can spread rapidly in cool, wet weather.",
        "symptoms": "Water-soaked dark patches on leaves, white fungal growth on undersides, rapid wilting.",
        "treatment": "Remove infected plants promptly; consult local advisory for approved fungicides immediately.",
        "prevention": "Use certified disease-free seed, avoid excess moisture, monitor during humid weather.",
    },
    "Potato___healthy": {
        "crop": "Potato", "disease": "Healthy",
        "description": "No significant disease symptoms detected.",
        "symptoms": "Leaf appears structurally normal.",
        "treatment": "No treatment required. Continue regular monitoring.",
        "prevention": "Maintain balanced fertilization and proper irrigation schedule.",
    },
    "Raspberry___healthy": {
        "crop": "Raspberry", "disease": "Healthy",
        "description": "No significant disease symptoms detected.",
        "symptoms": "Leaf appears structurally normal.",
        "treatment": "No treatment required. Continue regular monitoring.",
        "prevention": "Maintain balanced fertilization and routine pest monitoring.",
    },
    "Soybean___healthy": {
        "crop": "Soybean", "disease": "Healthy",
        "description": "No significant disease symptoms detected.",
        "symptoms": "Leaf appears structurally normal.",
        "treatment": "No treatment required. Continue regular monitoring.",
        "prevention": "Maintain balanced fertilization and proper irrigation schedule.",
    },
    "Squash___Powdery_mildew": {
        "crop": "Squash", "disease": "Powdery Mildew",
        "description": "A fungal disease producing a white powdery coating on leaves.",
        "symptoms": "White powdery patches on leaf surfaces and stems.",
        "treatment": "Improve air circulation; consult local advisory for approved fungicides.",
        "prevention": "Avoid excess nitrogen, ensure adequate plant spacing, water at soil level.",
    },
    "Strawberry___Leaf_scorch": {
        "crop": "Strawberry", "disease": "Leaf Scorch",
        "description": "A fungal disease causing purplish blotches and scorched-looking leaves.",
        "symptoms": "Small purple spots merging into larger scorched-looking blotches.",
        "treatment": "Remove infected leaves after harvest; consult local advisory for fungicide options.",
        "prevention": "Avoid overhead irrigation, ensure good airflow, remove old infected foliage.",
    },
    "Strawberry___healthy": {
        "crop": "Strawberry", "disease": "Healthy",
        "description": "No significant disease symptoms detected.",
        "symptoms": "Leaf appears structurally normal.",
        "treatment": "No treatment required. Continue regular monitoring.",
        "prevention": "Maintain balanced fertilization and routine pest monitoring.",
    },
    "Tomato___Bacterial_spot": {
        "crop": "Tomato", "disease": "Bacterial Spot",
        "description": "A bacterial disease causing small dark spots on leaves and fruit.",
        "symptoms": "Small water-soaked spots turning dark brown/black, leaf yellowing.",
        "treatment": "Remove infected plants/debris; consult local advisory for approved bactericide.",
        "prevention": "Use certified disease-free seed, avoid overhead watering, rotate crops.",
    },
    "Tomato___Early_blight": {
        "crop": "Tomato", "disease": "Early Blight",
        "description": "A common fungal disease causing dark concentric spots on lower leaves.",
        "symptoms": "Brown/black concentric ring spots on older leaves, yellowing around spots, leaf drop.",
        "treatment": "Remove and destroy affected leaves; improve air circulation; consider a suitable fungicide.",
        "prevention": "Crop rotation, avoid overhead watering, use disease-resistant varieties, maintain spacing.",
    },
    "Tomato___Late_blight": {
        "crop": "Tomato", "disease": "Late Blight",
        "description": "A serious disease that can spread rapidly in cool, wet weather.",
        "symptoms": "Water-soaked dark patches on leaves, white fungal growth on undersides, rapid wilting.",
        "treatment": "Remove infected plants promptly; consult local advisory for approved fungicides immediately.",
        "prevention": "Use certified disease-free seed, avoid excess moisture, monitor during humid weather.",
    },
    "Tomato___Leaf_Mold": {
        "crop": "Tomato", "disease": "Leaf Mold",
        "description": "A fungal disease favored by high humidity, common in greenhouse tomatoes.",
        "symptoms": "Pale green/yellow spots on upper leaf surface, olive-green mold on the underside.",
        "treatment": "Improve ventilation and reduce humidity; consult local advisory for fungicide options.",
        "prevention": "Avoid overhead watering, increase plant spacing, ensure good airflow.",
    },
    "Tomato___Septoria_leaf_spot": {
        "crop": "Tomato", "disease": "Septoria Leaf Spot",
        "description": "A fungal disease producing many small circular spots on leaves.",
        "symptoms": "Numerous small circular spots with dark borders and gray centers, mainly on lower leaves.",
        "treatment": "Remove affected leaves; consult local advisory for fungicide recommendations.",
        "prevention": "Rotate crops, avoid overhead watering, remove plant debris after harvest.",
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "crop": "Tomato", "disease": "Spider Mite Damage",
        "description": "Pest damage from tiny spider mites feeding on leaf tissue.",
        "symptoms": "Fine stippled/speckled yellowing on leaves, fine webbing in severe infestations.",
        "treatment": "Increase humidity around plants; consult local advisory for approved miticide if severe.",
        "prevention": "Regularly inspect leaf undersides, avoid drought stress, encourage natural predators.",
    },
    "Tomato___Target_Spot": {
        "crop": "Tomato", "disease": "Target Spot",
        "description": "A fungal disease producing concentric ringed lesions.",
        "symptoms": "Brown lesions with concentric rings resembling a target, on leaves and fruit.",
        "treatment": "Remove affected leaves; consult local advisory for fungicide recommendations.",
        "prevention": "Improve airflow, avoid overhead watering, rotate crops.",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "crop": "Tomato", "disease": "Yellow Leaf Curl Virus",
        "description": "A viral disease spread by whiteflies, causing stunted growth.",
        "symptoms": "Upward curling and yellowing of leaves, stunted plant growth, reduced fruit yield.",
        "treatment": "Remove and destroy infected plants; control whitefly populations; consult local advisory.",
        "prevention": "Use virus-resistant varieties, control whiteflies, use reflective mulches.",
    },
    "Tomato___Tomato_mosaic_virus": {
        "crop": "Tomato", "disease": "Mosaic Virus",
        "description": "A viral disease causing mottled leaf discoloration.",
        "symptoms": "Light and dark green mottled pattern on leaves, leaf distortion, stunted growth.",
        "treatment": "Remove and destroy infected plants; disinfect tools between plants; consult local advisory.",
        "prevention": "Use certified virus-free seed, wash hands/tools between handling plants, control aphids.",
    },
    "Tomato___healthy": {
        "crop": "Tomato", "disease": "Healthy",
        "description": "No significant disease symptoms detected.",
        "symptoms": "Leaf appears structurally normal.",
        "treatment": "No treatment required. Continue regular monitoring.",
        "prevention": "Maintain balanced fertilization and proper irrigation schedule.",
    },
}

GENERIC_FALLBACK = {
    "crop": "Unknown", "disease": "Unrecognized Condition",
    "description": "The model returned a class that is not in our lookup table.",
    "symptoms": "Not available for this class.",
    "treatment": "Please consult a local agricultural officer for an accurate diagnosis.",
    "prevention": "General good practice: crop rotation, proper spacing, and balanced fertilization.",
}


def get_disease_info(raw_label):
    """Look up friendly info for a raw model class label, with safe fallback.

    Different plant-disease models on Hugging Face use different label
    formats. This first tries an exact match against the PlantVillage-style
    keys above; if that fails, it tries to parse common alternate formats
    like "Tomato with Early Blight" or "Tomato - Early Blight" so the crop
    and disease names still display correctly even without a full lookup
    entry, before falling back to a fully generic response.
    """
    if raw_label in DISEASE_INFO:
        return DISEASE_INFO[raw_label]

    import re
    match = re.split(r'\s+with\s+|\s*-\s*|___', raw_label, maxsplit=1, flags=re.IGNORECASE)
    if len(match) == 2:
        crop, disease = match[0].strip(), match[1].strip().replace('_', ' ')
        is_healthy = disease.lower() in ('healthy', 'health')
        return {
            "crop": crop,
            "disease": "Healthy" if is_healthy else disease,
            "description": "No significant disease symptoms detected." if is_healthy else f"The AI model detected signs consistent with {disease.lower()} on this {crop.lower()} leaf.",
            "symptoms": "Leaf appears structurally normal." if is_healthy else "Symptoms vary — consult a local agricultural officer to confirm visually.",
            "treatment": "No treatment required. Continue regular monitoring." if is_healthy else "Consult a local agricultural officer or extension service to confirm this diagnosis before applying any treatment.",
            "prevention": "Maintain balanced fertilization and routine monitoring." if is_healthy else "General good practice: crop rotation, proper spacing, balanced fertilization, and routine field monitoring.",
        }

    return {**GENERIC_FALLBACK, "disease": raw_label}
