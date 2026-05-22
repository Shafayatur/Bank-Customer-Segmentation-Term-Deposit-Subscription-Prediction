Built an end-to-end data science project on bank customer segmentation and term deposit subscription prediction 🏦

Started with 41,188 customer records from the UCI Bank Marketing dataset and built a full pipeline:

📊 Exploratory Data Analysis across demographics, financial behavior, and campaign patterns
🤖 3 classification models — Logistic Regression, Random Forest, Decision Tree — all handling class imbalance with balanced weights (only 11.3% of customers subscribed)
🔵 KMeans clustering to segment customers into 3 actionable groups for targeted marketing
📈 2 interactive dashboards built in Streamlit and Dash by Plotly — upload any CSV and they update automatically

Key findings:
→ Senior customers (60+) subscribe at 45.5% — 4× the average
→ Cellular contact converts nearly 3× better than telephone
→ Previous campaign success predicts 65% re-subscription rate
→ Best ROC-AUC: ~0.800 (more reliable than accuracy on imbalanced data)

I used Claude (Anthropic) as an AI assistant throughout — for code review, catching bugs, and dashboard development. All analytical decisions, interpretations, and project direction were my own. Transparent AI use is something I think matters.
