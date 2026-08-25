# Credit Card Fraud Detection

سیستم تشخیص تقلب کارت اعتباری — یک پروژه‌ی یادگیری ماشین برای طبقه‌بندی تراکنش‌ها به سالم/تقلبی، با تمرکز ویژه بر مدیریت داده‌ی به‌شدت نامتوازن، جلوگیری از Data Leakage، و تنظیم اصولی Trade-off بین Precision و Recall.

## ساختار پروژه

```
Fraud-detection/
├── data/
│   └── creditcard.csv          # دیتاست خام (باید جداگانه دانلود شود، در گیت نیست)
├── models/
│   ├── model.pkl                # مدل نهایی (Logistic Regression) — با train.py ساخته می‌شود
│   ├── scaler.pkl                # StandardScaler فیت‌شده روی کل داده
│   └── model_config.json         # Threshold تصمیم‌گیری نهایی
├── reports/
│   ├── experiments.md            # گزارش کامل آزمایش‌ها، نتایج، و تحلیل
│   └── assets/                   # نمودارها (Boxplot پایداری، منحنی max_depth، منحنی PR)
├── src/
│   ├── data_prep.ipynb           # نوت‌بوک کامل: EDA، پیش‌پردازش، آموزش، آزمایش‌ها
│   ├── train.py                  # آموزش مدل نهایی از صفر و ذخیره‌ی آرتیفکت‌ها
│   ├── predict.py                # پیش‌بینی خط‌فرمانی (ورودی/خروجی JSON تک‌رکوردی)
│   ├── app.py                    # سرویس API (FastAPI) برای پیش‌بینی
│   └── test_api.py               # تست‌های Smoke برای app.py
└── README.md
```

## نصب

```bash
pip install pandas numpy scikit-learn joblib fastapi uvicorn pytest httpx
```

## ۱. آموزش مدل

دیتاست خام (`creditcard.csv`) باید در مسیر `data/creditcard.csv` قرار داشته باشد ([منبع دیتاست](https://www.kaggle.com/mlg-ulb/creditcardfraud)). سپس:

```bash
cd src
python train.py
```

این دستور:
- داده را می‌خواند، رکوردهای تکراری را حذف می‌کند
- `StandardScaler` را روی کل داده فیت می‌کند
- `LogisticRegression` را روی کل داده آموزش می‌دهد
- `model.pkl`, `scaler.pkl`, `model_config.json` را در `models/` ذخیره می‌کند

## ۲. پیش‌بینی از خط فرمان

ورودی، یک فایل JSON تک‌رکوردی شامل `Time`, `V1`...`V28`, `Amount` است:

```bash
cd src
python predict.py input.json
```

خروجی هم در ترمینال چاپ می‌شود و هم در `output.json` ذخیره می‌شود:

```json
{
  "prediction": "Fraud",
  "class_id": 1,
  "probability": 0.87,
  "threshold": 0.1168,
  "status": "success"
}
```

## ۳. اجرای API

```bash
cd src
uvicorn app:app --reload
```

سرور روی `http://127.0.0.1:8000` بالا می‌آید. مستندات تعاملی (Swagger UI) در `http://127.0.0.1:8000/docs` قابل مشاهده و تست است.

نمونه‌ی درخواست:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @input.json
```

### اجرای تست‌ها

```bash
cd src
pytest test_api.py -v
```

## نتایج خلاصه

| مدل نهایی | Precision | Recall | F1 |
|---|---:|---:|---:|
| Logistic Regression (Threshold=۰.۱۱۶۸) | ۰.۸۳۳ | ۰.۷۳۷ | ۰.۷۸۲ |

شرح کامل روش‌شناسی، سه آزمایش کنترل‌شده (اثر Scaling، اثر عمق درخت، تنظیم Threshold)، مقایسه‌ی مدل‌ها، و تفسیر کسب‌وکاری در [`reports/experiments.md`](./reports/experiments.md) آمده است.

## نکات مهم طراحی

- **جلوگیری از Data Leakage:** حذف تکراری‌ها قبل از Split، Scaler درون Pipeline (فیت فقط روی هر Fold از Train)، و انتخاب Threshold صرفاً از روی احتمالات out-of-fold مربوط به Train.
- **مدیریت عدم توازن کلاس:** به‌جای `class_weight` (که Precision را به‌شدت قربانی می‌کرد)، از تنظیم دقیق Threshold روی منحنی Precision-Recall استفاده شد.
- **جداسازی مدل و Scaler:** برخلاف Pipeline واحد استفاده‌شده در نوت‌بوک آزمایش، در محیط Production (`train.py`/`predict.py`/`app.py`) مدل و Scaler به‌صورت مجزا ذخیره و اعمال می‌شوند.