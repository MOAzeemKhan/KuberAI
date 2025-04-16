// --- Updated script.js with loading indicators and status checking ---
document.addEventListener("DOMContentLoaded", () => {
  const chatForm = document.getElementById("chat-form");
  const chatInput = document.getElementById("chat-input");
  const chatBox = document.getElementById("chat-box");
  const uploadBtn = document.getElementById("upload-btn");
  const fileInput = document.getElementById("file-input");

  let currentTaskId = null;
  let statusCheckInterval = null;

  // Chat submit handler
  chatForm.addEventListener("submit", async function (e) {
    e.preventDefault();

    const userMessage = chatInput.value.trim();
    if (userMessage === "") return;

    appendMessage("You", userMessage, "user");
    chatInput.value = "";

    // Show loading indicator
    const loadingId = showLoading();

    try {
      const response = await fetch("/api/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ question: userMessage })
      });

      // Remove loading indicator
      removeLoading(loadingId);

      const data = await response.json();
      if (data.status === "success") {
        appendMessage("KuberAI", data.answer, "bot");
      } else {
        appendMessage("KuberAI", `Error: ${data.message || "I couldn't understand that."}`, "bot");
      }
    } catch (error) {
      // Remove loading indicator
      removeLoading(loadingId);
      appendMessage("KuberAI", "Error reaching the backend. Please try again later.", "bot");
    }
  });

  // File upload trigger
  uploadBtn.addEventListener("click", () => {
    fileInput.click();
  });

  fileInput.addEventListener("change", async () => {
    const file = fileInput.files[0];
    if (!file) return;

    // Check if it's a PDF
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      appendMessage("KuberAI", "Please upload a PDF file only.", "bot");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    appendMessage("You", `📄 Uploaded: ${file.name}`, "user");

    // Show loading indicator
    const loadingId = showLoading();

    try {
      const response = await fetch("/upload_pdf", {
        method: "POST",
        body: formData
      });

      const result = await response.json();

      if (result.status === "processing" && result.task_id) {
        // Start checking status
        currentTaskId = result.task_id;
        appendMessage("KuberAI", "Processing your PDF... This may take a minute.", "bot");

        // Clear any existing interval
        if (statusCheckInterval) clearInterval(statusCheckInterval);

        // Check status every 3 seconds
        statusCheckInterval = setInterval(() => checkProcessingStatus(loadingId), 3000);
      } else {
        // Remove loading indicator
        removeLoading(loadingId);
        appendMessage("KuberAI", "Failed to process PDF. Please try again.", "bot");
      }
    } catch (err) {
      // Remove loading indicator
      removeLoading(loadingId);
      appendMessage("KuberAI", "Error uploading file. Please try again.", "bot");
    }
  });

  // Check processing status
  async function checkProcessingStatus(loadingId) {
    if (!currentTaskId) {
      clearInterval(statusCheckInterval);
      removeLoading(loadingId);
      return;
    }

    try {
      const response = await fetch(`/check_processing/${currentTaskId}`);
      const result = await response.json();

      if (result.status === "complete") {
        clearInterval(statusCheckInterval);
        removeLoading(loadingId);
        appendMessage("KuberAI", "✅ PDF processed successfully! You can now ask me questions about the document.", "bot");
        currentTaskId = null;
      } else if (result.status === "error") {
        clearInterval(statusCheckInterval);
        removeLoading(loadingId);
        appendMessage("KuberAI", `❌ ${result.message || "Error processing PDF."}`, "bot");
        currentTaskId = null;
      }
      // If still processing, continue waiting

    } catch (err) {
      console.error("Error checking status:", err);
      // Don't clear interval, try again
    }
  }

  // Helper functions for loading indicators
  function showLoading() {
    const loadingId = `loading-${Date.now()}`;
    const loadingElement = document.createElement("div");
    loadingElement.id = loadingId;
    loadingElement.className = "message bot loading";
    loadingElement.innerHTML = `
      <div class="loading-dots">
        <span class="dot"></span>
        <span class="dot"></span>
        <span class="dot"></span>
      </div>
    `;
    chatBox.appendChild(loadingElement);
    chatBox.scrollTop = chatBox.scrollHeight;
    return loadingId;
  }

  function removeLoading(loadingId) {
    const loadingElement = document.getElementById(loadingId);
    if (loadingElement) {
      loadingElement.remove();
    }
  }

  function appendMessage(sender, message, type) {
    const bubble = document.createElement("div");
    bubble.className = `message ${type}`;
    bubble.innerHTML = `<strong>${sender}:</strong> ${message}`;
    chatBox.appendChild(bubble);
    chatBox.scrollTop = chatBox.scrollHeight;
  }
});