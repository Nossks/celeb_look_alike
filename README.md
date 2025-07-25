## Celebrity Look‑Alike:-
A Flask‑powered web app that lets you upload a photo, automatically detects and crops your face, computes a 128‑dimensional face embedding (via a pretrained ResNet50 + custom L2‑normalization layer), and finds the closest celebrity look‑alike using cosine similarity.

## Features:-
**Face Detection & Cropping**  
  Uses MediaPipe Face Detection to locate and crop the primary face to 256 × 256 pixels.
**Embedding Model**  
  ResNet50 backbone + small MLP head → 128‑dim embeddings, then L2‑normalized via a custom `L2NormLayer`.
  then, the model is fine tuned and trained for more epochs
**Similarity Matching**  
  Precomputed celebrity embeddings stored in `data_embedding.pkl`. Cosine similarity finds the top match.
**Interactive UI**  
  Upload your photo, see the cropped face preview, and read “You look like …” with a similarity score.
**Clean Responsive Design**  
  Minimal HTML + modern CSS (hover effects, gradients, responsive container).

## Project Structure:-
celeb_look_alike/
├── app.py # Flask backend (face crop, embed, match)
├── face_embedding_model.keras # Saved embedding model
├── data_embedding.pkl # Pickled {celebrity_name: 128‑dim array}
├── requirements.txt # Python dependencies
├── static/
│ ├── style.css # App styles
│ └── uploads/ # (ephemeral) cropped face images
├── templates/
│ └── index.html # Main upload & result page
├── model_training.ipynb # Notebook to build & save the embedding model
├── create_dataset.ipynb # Notebook to extract celeb faces & embeddings
└── README.md # Project overview (this file)

## How It Works
1. **Upload** — User selects an image file in the browser form.  
2. **Detect & Crop** — Server reads the bytes, runs MediaPipe to find the face, crops and resizes to 256 × 256.  
3. **Embed** — The cropped face is passed through `face_embedding_model` to get a 128‑dim vector.  
4. **Match** — Compute cosine similarity against every celeb embedding; pick the highest score.  
5. **Display** — Render the cropped face preview and the text result on the same page.

## Running Locally
# Clone the repo
git clone https://github.com/Nossks/celeb_look_alike.git
cd celeb_look_alike
# Create & activate virtualenv
python3 -m venv venv
source venv/bin/activate        
# Install dependencies
pip install -r requirements.txt
# Run the app
python app.py