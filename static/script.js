document.addEventListener("DOMContentLoaded", () => {
  // Sidebar Toggle
  const sidebar = document.getElementById("sidebar");
  const toggleBtn = document.querySelector(".toggle-btn");

  toggleBtn.addEventListener("click", () => {
    sidebar.classList.toggle("collapsed");

    const labels = document.querySelectorAll(".nav-item .label");
    labels.forEach(label => {
      label.style.display = sidebar.classList.contains("collapsed") ? "none" : "inline";
    });
  });

  // Slider Value Display
  const slider = document.getElementById("amount-slider");
  const sliderVal = document.getElementById("slider-val");

  slider.addEventListener("input", () => {
    sliderVal.textContent = "₹" + Number(slider.value).toLocaleString();
  });

  // Fetch Recommendations
  const getBtn = document.getElementById("get-recommendations");
  const resultContainer = document.getElementById("recommendation-list");

  getBtn.addEventListener("click", async () => {
    try {
      resultContainer.innerHTML = "<p>Loading...</p>";

      const response = await fetch("/api/recommend");
      const result = await response.json();

      if (result.status === "success") {
        resultContainer.innerHTML = "";

        result.data.forEach((rec) => {
          const item = document.createElement("div");
          item.className = "stock-card";
          item.innerHTML = `
            <div><strong>${rec.ticker.toUpperCase()}</strong> (${rec.recommendation})</div>
            <div class="confidence-bar">
              <div class="confidence-fill" style="width: ${rec.allocation}%;"></div>
            </div>
            <div class="explanation">Positive RSI • Low volatility • Strong sentiment</div>
          `;
          resultContainer.appendChild(item);
        });
      } else {
        resultContainer.innerHTML = `<p>⚠️ ${result.message}</p>`;
      }
    } catch (err) {
      console.error("Error fetching recommendations:", err);
      resultContainer.innerHTML = `<p>Failed to fetch recommendations.</p>`;
    }
  });

  // Chatbot Integration
  const chatForm = document.getElementById("chat-form");
  const chatInput = document.getElementById("chat-input");
  const chatBox = document.getElementById("chat-box");

  chatForm?.addEventListener("submit", async (e) => {
    e.preventDefault();

    const userMessage = chatInput.value.trim();
    if (!userMessage) return;

    appendMessage("You", userMessage, "user");
    chatInput.value = "";

    try {
      const response = await fetch("/api/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ question: userMessage })
      });

      const data = await response.json();
      if (data.status === "success") {
        appendMessage("KuberAI", data.answer, "bot");
      } else {
        appendMessage("KuberAI", "Sorry, I couldn’t understand that.", "bot");
      }
    } catch (error) {
      appendMessage("KuberAI", "Something went wrong! Please try again.", "bot");
    }
  });

  function appendMessage(sender, message, type) {
    const bubble = document.createElement("div");
    bubble.className = `message ${type}`;
    bubble.innerHTML = `<strong>${sender}:</strong> ${message}`;
    chatBox.appendChild(bubble);
    chatBox.scrollTop = chatBox.scrollHeight;
  }
});
