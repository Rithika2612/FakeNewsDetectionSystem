document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("news-form");
  const input = document.getElementById("news-input");
  const resultBox = document.getElementById("results");
  const loader = document.getElementById("loader");
  const buttonText = document.querySelector(".button-text");
  const newsFeed = document.getElementById("news-feed");

  // -------------------
  // Load latest news
  // -------------------
  async function loadLatestNews() {
    try {
      const response = await fetch("/get_latest_news");
      const data = await response.json();
      if (data.error) {
        newsFeed.innerHTML = `<li style="color:red;">${data.error}</li>`;
        return;
      }
      newsFeed.innerHTML = data.map(item =>
        `<li style="margin-bottom:10px;">
           <a href="${item.link}" target="_blank" style="color:#81c784;">${item.title}</a>
         </li>`).join('');
    } catch (err) {
      newsFeed.innerHTML = `<li style="color:red;">Failed to load news</li>`;
    }
  }
  // Call it when page loads
  loadLatestNews();
  setInterval(loadLatestNews, 3600);

  // -------------------
  // Prediction form
  // -------------------
  form.addEventListener("submit", async function (e) {
    e.preventDefault(); // stop default form behavior

    const text = input.value.trim();
    if (!text) {
      showError("⚠️ Please enter some news text before submitting.");
      return;
    }

    loader.style.display = "inline-block";
    buttonText.style.display = "none";

    const formData = new FormData();
    formData.append("news_text", text);

    try {
      const response = await fetch("/predict", {
        method: "POST",
        body: formData
      });

      let data;
      try {
        data = await response.json();
      } catch {
        showError(" Server returned HTML instead of JSON. Check Flask logs.");
        return;
      }

      if (!response.ok) {
        showError(data.error || " Prediction failed.");
        return;
      }

      // Display results
      resultBox.innerHTML = `
        <h3>Prediction Results</h3>
        <hr>
        <h5><strong>Logistic Regression:</strong> ${data.LogisticRegression === 1 ? "Real" : "Fake"}</h5>
        <h5><strong>Decision Tree:</strong> ${data.DecisionTree === 1 ? "Real" : "Fake"}</h5>
        <h5><strong>Random Forest:</strong> ${data.RandomForest === 1 ? "Real" : "Fake"}</h5>
        <h5><strong>Gradient Boosting:</strong> ${data.GradientBoosting === 1 ? "Real" : "Fake"}</h5>
        <h5><strong>BERT:</strong> ${data.BERT === 1 ? "Real" : "Fake"}</h5>
        <hr>
        <h2 style="color:${data.FinalResult === "Real" ? "green" : "red"}">
          Final Result: ${data.FinalResult}
        </h2>
      `;
    } catch (err) {
      showError("⚠️ " + err.message);
    } finally {
      loader.style.display = "none";
      buttonText.style.display = "inline";
    }
  });

  function showError(msg) {
    resultBox.innerHTML = `<div style="color:red;font-weight:bold;margin-top:10px;">${msg}</div>`;
    loader.style.display = "none";
    buttonText.style.display = "inline";
  }

  // Logout handler
  window.logout = function () {
    window.location.href = "/logout";
  }
});
