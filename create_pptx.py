from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

DARK_BG = RGBColor(0x1E, 0x1E, 0x2E)
ACCENT = RGBColor(0x89, 0xB4, 0xFA)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GRAY = RGBColor(0xBA, 0xC2, 0xDE)


def set_bg(slide):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = DARK_BG


def add_title_slide(title, subtitle):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    txBox = slide.shapes.add_textbox(Inches(1), Inches(2.5), Inches(11), Inches(2))
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(44)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.alignment = PP_ALIGN.CENTER

    p2 = tf.add_paragraph()
    p2.text = subtitle
    p2.font.size = Pt(22)
    p2.font.color.rgb = GRAY
    p2.alignment = PP_ALIGN.CENTER
    return slide


def add_content_slide(title, bullets):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    # Title
    txBox = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11), Inches(1))
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = ACCENT
    # Bullets
    txBox2 = slide.shapes.add_textbox(Inches(1), Inches(1.5), Inches(11), Inches(5.5))
    tf2 = txBox2.text_frame
    tf2.word_wrap = True
    for i, bullet in enumerate(bullets):
        p = tf2.paragraphs[0] if i == 0 else tf2.add_paragraph()
        p.text = bullet
        p.font.size = Pt(20)
        p.font.color.rgb = WHITE
        p.space_after = Pt(10)
    return slide


def add_two_col_slide(title, left_items, right_items):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    txBox = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11), Inches(1))
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = ACCENT

    # Left column
    left = slide.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(5.5), Inches(5.5))
    tf_l = left.text_frame
    tf_l.word_wrap = True
    for i, item in enumerate(left_items):
        p = tf_l.paragraphs[0] if i == 0 else tf_l.add_paragraph()
        p.text = item
        p.font.size = Pt(18)
        p.font.color.rgb = WHITE
        p.space_after = Pt(8)

    # Right column
    right = slide.shapes.add_textbox(Inches(6.8), Inches(1.5), Inches(5.5), Inches(5.5))
    tf_r = right.text_frame
    tf_r.word_wrap = True
    for i, item in enumerate(right_items):
        p = tf_r.paragraphs[0] if i == 0 else tf_r.add_paragraph()
        p.text = item
        p.font.size = Pt(18)
        p.font.color.rgb = WHITE
        p.space_after = Pt(8)
    return slide


# === SLIDES ===

# 1. Title
add_title_slide(
    "📊 Data Processing Pipeline",
    "Hackathon Project — Interactive ML Data Conditioning Platform"
)

# 2. Problem
add_content_slide("The Problem", [
    "• Data scientists spend 80% of time on data preparation",
    "• No single tool covers the full preprocessing catalog",
    "• Operations are scattered across notebooks — not reproducible",
    "• Teams repeat the same cleaning steps project after project",
    "",
    "💡 We built an interactive, all-in-one data conditioning app",
])

# 3. Solution
add_content_slide("Our Solution", [
    "• Streamlit-based web app — zero setup for end users",
    "• Upload any CSV/Excel/TSV → instant analysis & processing",
    "• 14 processing categories, 64+ operations",
    "• Stateful pipeline — chain operations, preview results live",
    "• One-click download of processed dataset",
    "• Reset to original at any point",
])

# 4. Architecture
add_content_slide("Architecture", [
    "📁 app.py — Streamlit UI (tabs: Overview, Stats, Viz, Filter, Process)",
    "📁 processing.py — Modular processing engine (64 functions)",
    "📁 sample_data.csv — Demo dataset with edge cases",
    "📁 test_processing.py — 64 automated tests (all passing ✅)",
    "",
    "Tech Stack:",
    "   • Python 3.14 + Streamlit",
    "   • pandas, numpy, scipy",
    "   • scikit-learn, imbalanced-learn",
    "   • Plotly for interactive visualizations",
])

# 5-6. Categories
add_two_col_slide(
    "14 Processing Categories",
    [
        "1. Data Quality",
        "   Missing values, duplicates",
        "2. Numeric Cleaning",
        "   Type conversion, precision",
        "3. Outlier Processing",
        "   Z-score, IQR, Isolation Forest, LOF",
        "4. Distribution Conditioning",
        "   Log, Sqrt, Box-Cox, Yeo-Johnson, Quantile",
        "5. Feature Scaling",
        "   Standard, MinMax, Robust, MaxAbs, UnitVec",
        "6. Categorical Processing",
        "   Label, One-Hot, frequency analysis",
        "7. Text Processing",
        "   Lowercase, whitespace, punctuation",
    ],
    [
        "8. Datetime Processing",
        "   Parse, extract year/month/day/weekday",
        "9. Feature Engineering",
        "   Ratios, interactions, polynomial, binning",
        "10. Time-Series Features",
        "   Lag, lead, rolling, EWM",
        "11. Statistical Analysis",
        "   Correlation, t-test, ANOVA, chi-square",
        "12. Feature Selection",
        "   Variance, MI, Tree importance, RFE, LASSO",
        "13. Dimensionality Reduction",
        "   PCA, Kernel PCA, ICA, FA, t-SNE",
        "14. Target Conditioning",
        "   SMOTE, ADASYN, over/under-sampling",
    ],
)

# 7. Demo highlights
add_content_slide("Key Features — Demo Highlights", [
    "🔄 Stateful Processing Pipeline",
    "   → Each operation updates the working dataset in session state",
    "",
    "📊 Data Quality Score (0-100)",
    "   → Completeness, Uniqueness, Consistency, Validity",
    "   → Color-coded: 🟢 ≥80  🟡 ≥60  🔴 <60",
    "",
    "📈 Before/After Compare Tab",
    "   → Side-by-side distribution plots, stats delta, quality score change",
    "",
    "🕐 Operation History",
    "   → Every action logged, full audit trail, reset anytime",
])

# 8. More highlights
add_content_slide("Key Features — continued", [
    "📊 Live Preview & Download",
    "   → See processed data instantly, export CSV anytime",
    "",
    "🧪 64 Automated Tests — All Passing",
    "   → Validates every function with edge cases (NaN, outliers, duplicates)",
    "",
    "🛡️ Robust Error Handling",
    "   → Auto-adjusts SMOTE k_neighbors, handles NaN in sklearn transforms",
    "",
    "🐳 Docker Support",
    "   → One command: docker compose up --build",
])

# 8. How to run
add_content_slide("How to Run", [
    "$ cd alpha_Hackathon",
    "$ python3 -m venv venv",
    "$ source venv/bin/activate",
    "$ pip install -r requirements.txt",
    "$ streamlit run app.py",
    "",
    "Then upload sample_data.csv (or any dataset) to start processing!",
])

# 9. What's next
add_content_slide("Future Enhancements", [
    "• Pipeline export — save & replay processing steps as JSON",
    "• AutoML integration — auto-suggest preprocessing per column type",
    "• UMAP & Autoencoder for dimensionality reduction",
    "• NLP: stopword removal, TF-IDF, embeddings",
    "• Data drift monitoring (Evidently integration)",
    "• Multi-user collaboration & dataset versioning (DVC)",
])

# 10. Thank you
add_title_slide(
    "Thank You! 🎉",
    "64 operations • 14 categories • 1 app\n\nQuestions?"
)

prs.save("presentation.pptx")
print("✅ presentation.pptx created!")
