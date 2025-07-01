# Credit Scoring Business Understanding

The Basel II Accord emphasizes the importance of robust risk measurement and management frameworks for financial institutions. For credit risk, this means banks must accurately assess and quantify their exposure to potential defaults. As a result, our credit scoring model must be interpretable and well-documented for several reasons:

### Regulatory Compliance and Validation

Basel II requires banks to validate their internal risk models. An interpretable model enables regulators and auditors to understand its logic, assumptions, and decision-making process. This transparency is essential for regulatory approval and demonstrating the model’s suitability.

### Risk Capital Allocation

Capital requirements are linked to a bank’s risk profile. A well-understood model ensures that risk capital is allocated appropriately. In contrast, “black box” models can lead to incorrect capital calculations, exposing the bank to unnecessary risk or inefficiency.

### Model Governance and Explainability

Banks must maintain clear governance over their models. Interpretability makes it easier to explain decisions to stakeholders—loan officers, management, and customers—building trust and ensuring consistent policy application.

### Monitoring and Recalibration

Understanding model drivers allows for effective monitoring and timely recalibration as market conditions or customer behavior change. Opaque models make diagnosing issues and maintaining performance more difficult.

### Auditability

Every step from data input to model output must be auditable. Clear documentation of features, transformations, and algorithms ensures the credit scoring process can be traced and verified.

---

#### Why Create a Proxy Variable for Default?

When a direct “default” label is unavailable—common with new services or evolving definitions—a proxy variable is needed. For example, in Bati Bank’s new buy-now-pay-later service, historical default data may not exist. Instead, we infer risk from behavioral data such as Recency, Frequency, and Monetary (RFM) patterns.

**Necessity of a Proxy Variable:**

- **Absence of Ground Truth:** Without a direct default flag, a proxy serves as a substitute target variable correlated with actual default.
- **Leveraging Available Data:** RFM data is available and can provide insights into customer risk.
- **Early Risk Identification:** A well-defined proxy helps identify risky customers early for proactive mitigation.

**Potential Business Risks:**

- **Proxy Misalignment:** If the proxy does not reflect true default, predictions may be inaccurate, leading to:
  - **High False Positives:** Denying credit to low-risk customers, causing lost revenue and dissatisfaction.
  - **High False Negatives:** Granting credit to high-risk customers, increasing loan losses.
- **Regulatory Scrutiny:** Regulators may question models built on unproven proxies, delaying launches or requiring extensive validation.
- **Reputational Damage:** Inaccurate assessments can harm the bank’s reputation.
- **Operational Inefficiencies:** Poor proxies may increase manual reviews or misclassifications.
- **Data/Concept Drift:** The relationship between RFM and default may change over time, requiring regular review.

---

#### Trade-offs: Simple vs. Complex Models in Regulated Finance

**Simple, Interpretable Models (e.g., Logistic Regression with WoE):**

- **Pros:**
  - High interpretability—feature impacts are clear.
  - Favored by regulators for transparency and auditability.
  - Robust and less prone to overfitting.
  - Easier to implement and monitor.
- **Cons:**
  - May have lower predictive performance.
  - Requires significant feature engineering.

**Complex, High-Performance Models (e.g., Gradient Boosting):**

- **Pros:**
  - High predictive accuracy—captures complex patterns.
  - Handles diverse data types and missing values.
  - Reduces manual feature engineering.
- **Cons:**
  - Low interpretability—difficult to explain decisions.
  - Regulatory challenges due to lack of transparency.
  - Higher risk of overfitting and more complex monitoring.
  - Greater computational cost.

**Key Trade-off:**  
In regulated finance, interpretability and regulatory acceptance often outweigh gains in predictive performance. While complex models may offer better accuracy, their lack of transparency can create regulatory, reputational, and operational risks. A common approach is to use interpretable models for core decisions and complex models for secondary insights, provided their outputs can be explained and validated.
