document.getElementById("churnForm").addEventListener("submit", async function(e) {
  e.preventDefault();

  const data = {
    gender: document.getElementById("gender").value,
    SeniorCitizen: parseInt(document.getElementById("SeniorCitizen").value),
    Partner: document.getElementById("Partner").value,
    Dependents: document.getElementById("Dependents").value,
    tenure: parseInt(document.getElementById("tenure").value),
    PhoneService: document.getElementById("PhoneService").value,
    InternetService: document.getElementById("InternetService").value,
    Contract: document.getElementById("Contract").value,
    PaperlessBilling: document.getElementById("PaperlessBilling").value,
    PaymentMethod: document.getElementById("PaymentMethod").value,
    MonthlyCharges: parseFloat(document.getElementById("MonthlyCharges").value),
    TotalCharges: parseFloat(document.getElementById("TotalCharges").value)
  };

  const response = await fetch("/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(data)
  });

  const result = await response.json();
  const display = result.prediction === 1
    ? `⚠️ The customer is likely to CHURN(LEAVE).\nConfidence: ${result.confidence * 100}%`
    : `✅ The customer is likely to STAY.\nConfidence: ${(1 - result.confidence) * 100}%`;

  document.getElementById("result").innerText = display;
});
