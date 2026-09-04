# AI Smart Agro — Real AI Server (No Demo Mode)

This is the external Python AI service referenced by the PHP website's
`includes/config.php`. It provides two **real, non-demo** endpoints:

- `POST /api/predict-disease` — real image classification using a free,
  publicly available model trained on the actual PlantVillage dataset
  (via Hugging Face's Inference API).
- `POST /api/recommend-crop` — a real, locally trained scikit-learn
  RandomForestClassifier (94–95% validation accuracy) that predicts the
  top 3 crops from soil/climate inputs.

InfinityFree cannot run Python, so this must be hosted **separately**,
on a free Python-friendly host. Render.com's free tier is used below,
but Railway.app or PythonAnywhere work too.

---

## 1. Get a free Hugging Face API token

1. Create a free account at huggingface.co
2. Go to **Settings → Access Tokens → New Token** (Read access is enough)
3. Copy the token (starts with `hf_...`) — you'll paste it as an
   environment variable in Step 3 below.

**Important — verify the model still exists before relying on it:**
This project defaults to the public model
`linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification`
(trained on the PlantVillage dataset, 38 disease classes). Public model
availability can change over time. Before deploying:
1. Visit `https://huggingface.co/<model-id>` in your browser to confirm
   it still exists and supports the free Inference API.
2. If it's gone, search huggingface.co for another `image-classification`
   model trained on "plant disease" or "PlantVillage" and update the
   `HF_MODEL_ID` environment variable to match (see Step 3).
3. If a class label returned by the new model isn't in
   `disease_info.py`'s `DISEASE_INFO` dictionary, the app safely falls
   back to a generic entry — but for best results, add entries for the
   new model's class names using the same file's format.

## 2. Test locally (optional but recommended)

```bash
cd ai-server
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python3 train_crop_model.py       # creates crop_model.joblib (real trained model)

export AI_API_KEY="choose-a-long-random-secret"
export HF_API_TOKEN="hf_your_token_here"
python3 app.py
```

Visit `http://localhost:5000/` — you should see a small JSON status message.

Test crop recommendation:
```bash
curl -X POST http://localhost:5000/api/recommend-crop \
  -H "X-API-KEY: choose-a-long-random-secret" \
  -H "Content-Type: application/json" \
  -d '{"nitrogen":90,"phosphorus":45,"potassium":45,"temperature":26,"humidity":80,"rainfall":200,"ph":6.2}'
```

Test disease detection (replace path with a real leaf photo):
```bash
curl -X POST http://localhost:5000/api/predict-disease \
  -H "X-API-KEY: choose-a-long-random-secret" \
  -F "image=@/path/to/leaf.jpg"
```

## 3. Deploy to Render (free tier)

1. Push this `ai-server` folder to its own GitHub repository.
2. Go to render.com → **New → Web Service** → connect your GitHub repo.
3. Settings:
   - **Build Command:** `pip install -r requirements.txt && python3 train_crop_model.py`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT`
   - **Environment Variables** (Render dashboard → Environment):
     - `AI_API_KEY` = a long random secret string you choose
     - `HF_API_TOKEN` = your Hugging Face token from Step 1
     - `HF_MODEL_ID` = (optional) only set this if you're using a
       different model than the default
4. Deploy. Render will give you a URL like
   `https://ai-smart-agro-ai.onrender.com`.

**Free tier note:** Render's free web services "sleep" after ~15 minutes
of inactivity and take 30-60 seconds to wake on the next request. Your
PHP site's `AI_API_TIMEOUT` (in `includes/config.php`) is currently set
to 8 seconds — you may want to raise it to 60 for the first request
after idle periods, or accept that the first request after a while
will fall back to Demo Mode automatically (which is a safe, intended behavior).

## 4. Point the PHP website at this server

In your InfinityFree site's `includes/config.php`, update:

```php
define('DEMO_MODE', false);
define('AI_DISEASE_API_URL', 'https://ai-smart-agro-ai.onrender.com/api/predict-disease');
define('AI_CROP_API_URL', 'https://ai-smart-agro-ai.onrender.com/api/recommend-crop');
define('AI_API_KEY', 'the-same-secret-you-set-as-AI_API_KEY-on-render');
define('AI_API_TIMEOUT', 60);
```

The `AI_API_KEY` value **must match exactly** between this Python server's
environment variable and the PHP site's `config.php` — this is the shared
secret that authenticates requests between them.

Re-upload `config.php` to InfinityFree, then test:
1. Log in to your farmer dashboard
2. Try **Disease Detection** with a real leaf photo — you should see a
   real predicted disease name and confidence score (no "Demo Prediction" badge)
3. Try **Crop Recommendation** — you should see real model probabilities
   (no "Demo Prediction – Rule-Based Fallback" badge)

## 5. Understanding the crop recommendation model's training data

`train_crop_model.py` trains on data generated from **documented
agronomic requirement ranges** for ~15 common crops (nitrogen/phosphorus/
potassium needs, ideal temperature, humidity, rainfall, and soil pH from
standard agronomy references) — not a downloaded historical dataset.
This is a genuinely trained ML model (RandomForest learns real decision
boundaries and gives probability-based predictions), but it is worth
being transparent about this in your project report/viva: the *ranges*
are real agronomic knowledge, but individual training rows are
synthetically sampled within those ranges rather than pulled from
recorded historical farm data.

**If you obtain a real historical dataset** (for example, the publicly
known "Crop Recommendation Dataset" with columns N, P, K, temperature,
humidity, ph, rainfall, label), save it as `ai-server/real_data.csv` and
re-run `python3 train_crop_model.py` — the script automatically detects
and prefers real data over the synthetic generator, producing an even
more defensible model for your report.

## 6. Costs

- Hugging Face free Inference API: free tier with rate limits (fine for
  a college project demo; heavy continuous use may hit limits).
- Render free web service: free, but sleeps when idle (see note above).
- No cost is required to run this exactly as configured.
