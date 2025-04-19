document.addEventListener("DOMContentLoaded", () => {
  const chatForm = document.getElementById("chat-form");
  const chatInput = document.getElementById("chat-input");
  const chatBox = document.getElementById("chat-box");
  const uploadBtn = document.getElementById("upload-btn");
  const fileInput = document.getElementById("file-input");
  const sidebar = document.getElementById('sidebar');
  const recommendationArea = document.getElementById('recommendation-area');
  const sidebarToggle = document.getElementById('sidebar-toggle');
  const recommendationToggle = document.getElementById('recommendation-toggle');
  const mainContainer = document.querySelector('.main');
  const recommendationList = document.getElementById("recommendation-list");
  const summaryBoxId = "portfolio-summary";

  // Restore saved selection
  const saved = localStorage.getItem("kuberai-recommendation");
  if (saved) {
    const data = JSON.parse(saved);
    document.getElementById("amount-slider").value = data.investment_amount;
    document.getElementById("slider-val").textContent = "₹" + data.investment_amount.toLocaleString();

    const riskBtn = [...document.querySelectorAll('.input-group:nth-of-type(2) .pill-button')]
      .find(btn => btn.innerText === data.risk_level);
    if (riskBtn) riskBtn.classList.add("active");

    const horizonBtn = [...document.querySelectorAll('.input-group:nth-of-type(3) .pill-button')]
      .find(btn => btn.innerText === data.horizon);
    if (horizonBtn) horizonBtn.classList.add("active");
  }

  // Initialize recommendation functionality
  if (document.getElementById("get-recommendations")) {
    document.getElementById("get-recommendations").addEventListener("click", async () => {
      // Get the selected values
      const risk = document.querySelector('.input-group:nth-of-type(2) .pill-button.active')?.innerText || 'Medium';
      const horizon = document.querySelector('.input-group:nth-of-type(3) .pill-button.active')?.innerText || 'Short Term';
      const amount = document.getElementById('amount-slider').value;

      console.log("Button clicked. Sending data to backend:", { risk, horizon, amount }); // Debugging log

      try {
        const res = await fetch("/api/recommendations", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ risk, horizon, amount })
        });

        const data = await res.json();
        console.log("Recommendation Data:", data); // Debugging line

        const { recommendations, explanation } = data;

        if (!recommendations || recommendations.length === 0) {
          recommendationList.innerHTML = "<p>No recommendations available.</p>";
          return;
        }

        // Clear old content
        recommendationList.innerHTML = "";

        // Save to localStorage
        localStorage.setItem("kuberai-recommendation", JSON.stringify(data));

        // Add summary if not exists
        let summaryBox = document.getElementById(summaryBoxId);
        if (!summaryBox) {
          summaryBox = document.createElement("div");
          summaryBox.id = summaryBoxId;
          summaryBox.style.marginTop = "20px";
          summaryBox.style.backgroundColor = "#fffbea";
          summaryBox.style.padding = "15px";
          summaryBox.style.borderRadius = "10px";
          summaryBox.style.fontSize = "14px";
          recommendationList.parentNode.appendChild(summaryBox);
        }

        // 🎴 Render Cards
        recommendations.forEach(rec => {
          const card = document.createElement("div");
          card.className = "recommendation-card";
          card.innerHTML = `
            <div class="card-header">
              <span class="ticker-pill">${rec.ticker.toUpperCase()}</span>
            </div>
            <div class="card-content">
              <p><strong>Amount to Invest:</strong> ₹${rec.amount_to_invest.toLocaleString()}</p>
              <p><strong>Action:</strong> ${rec.recommendation}</p>
            </div>
          `;
          recommendationList.appendChild(card);
        });

        // 🧠 Set Explanation
        summaryBox.innerText = explanation;

      } catch (err) {
        recommendationList.innerHTML = "<p style='color:red'>Failed to fetch recommendations. Please try again.</p>";
        console.error("Error fetching recommendations:", err);
      }
    });
  }

  // 📄 CSV Download
  if (document.getElementById("download-csv")) {
    document.getElementById("download-csv").addEventListener("click", () => {
      const saved = localStorage.getItem("kuberai-recommendation");
      if (!saved) return alert("No recommendations to export.");

      const data = JSON.parse(saved);
      const rows = [
        ["Ticker", "Amount to Invest", "Recommendation"],
        ...data.recommendations.map(r => [r.ticker, r.amount_to_invest, r.recommendation])
      ];

      const csv = rows.map(row => row.join(",")).join("\n");
      const blob = new Blob([csv], { type: "text/csv" });
      const url = URL.createObjectURL(blob);

      const a = document.createElement("a");
      a.href = url;
      a.download = "kuberai_recommendations.csv";
      a.click();
      URL.revokeObjectURL(url);
    });
  }

  // 🧾 PDF Download
  if (document.getElementById("download-pdf")) {
    document.getElementById("download-pdf").addEventListener("click", () => {
      const saved = localStorage.getItem("kuberai-recommendation");
      if (!saved) return alert("No recommendations to export.");

      const { recommendations, explanation, risk_level, horizon, investment_amount } = JSON.parse(saved);
      const { jsPDF } = window.jspdf;

      const doc = new jsPDF();

      // Title
      doc.setFont("helvetica", "bold");
      doc.setFontSize(20);
      doc.text("KuberAI Investment Recommendations", 15, 20);

      // Basic Info
      doc.setFontSize(12);
      doc.setFont("helvetica", "normal");
      doc.text(`Investment Amount: ₹${investment_amount.toLocaleString()}`, 15, 32);
      doc.text(`Risk Level: ${risk_level}`, 15, 40);
      doc.text(`Horizon: ${horizon}`, 15, 48);

      // Section: Top 5 Recommendations
      doc.setFontSize(14);
      doc.setFont("helvetica", "bold");
      doc.text("Top 5 Recommendations:", 15, 60);

      let y = 70;
      doc.setFontSize(12);
      doc.setFont("helvetica", "normal");

      recommendations.forEach((r, idx) => {
        doc.text(`${idx + 1}. ${r.ticker.toUpperCase()} — ₹${r.amount_to_invest.toLocaleString()} — ${r.recommendation}`, 15, y);
        y += 8;
      });

      // Section: Portfolio Summary
      y += 8;
      doc.setFont("helvetica", "bold");
      doc.setFontSize(14);
      doc.text("Portfolio Summary:", 15, y);
      y += 10;

      doc.setFont("helvetica", "normal");
      doc.setFontSize(11);

      const explanationLines = doc.splitTextToSize(explanation, 180); // Fit inside margins
      doc.text(explanationLines, 15, y);

      // Save
      doc.save("KuberAI_Recommendations.pdf");
    });
  }

  // Sidebar Toggle
  if (sidebarToggle) {
    sidebarToggle.addEventListener('click', () => {
      sidebar.classList.toggle('collapsed');
      document.querySelector('.main').classList.toggle('sidebar-collapsed');
    });
  }

  // Recommendation Panel Toggle
  if (recommendationToggle) {
    recommendationToggle.addEventListener('click', () => {
      recommendationArea.classList.toggle('collapsed');
      mainContainer.classList.toggle('recommendation-collapsed');
      document.querySelector('.main').classList.toggle('recommendation-collapsed');
    });
  }

  // Initialize page navigation
  initPageNavigation();

  // Initialize slideable panels and toggle buttons
  initSlidablePanels();

  let currentTaskId = null;
  let statusCheckInterval = null;
  let isDarkMode = localStorage.getItem("kuberai-dark-mode") === "true";

  // Apply dark mode if saved
  if (isDarkMode) {
    document.body.classList.add("dark-mode");
    const darkModeToggle = document.getElementById("dark-mode-toggle");
    if (darkModeToggle) {
      darkModeToggle.innerHTML = '<i class="fas fa-sun"></i> Light Mode';
    }
  }

  // Load saved chat history from localStorage if on chat page
  if (chatBox) {
    loadChatHistory();
  }

  // Initialize UI controls
  setupUIControls();

  // Set up event listeners
  setupEventListeners();

  // Add emoji picker and voice input if on chat page
  if (chatForm) {
    setupEmojiAndVoiceInput();
  }

  // Initialize page-specific features
  initPageSpecificFeatures();

  // Initialize page navigation
  function initPageNavigation() {
    const navItems = document.querySelectorAll('.nav-item');

    navItems.forEach(item => {
      item.addEventListener('click', () => {
        const page = item.getAttribute('data-page');
        if (page) {
          // Store current page in localStorage
          localStorage.setItem('kuberai-current-page', page);

          // Navigate to the page
          if (page === 'index') {
            window.location.href = '/';
          } else {
            window.location.href = `/${page}`;
          }
        } else if (item.classList.contains('logout')) {
          // Handle logout action
          if (confirm("Are you sure you want to log out?")) {
            // In a real app this would call a logout endpoint
            window.location.href = '/login';
          }
        }
      });
    });
  }

  // Initialize page-specific features
  function initPageSpecificFeatures() {
    const currentPage = window.location.pathname;

    // Profile Page Features
    if (currentPage.includes('profile')) {
      initProfileFeatures();
    }

    // Settings Page Features
    if (currentPage.includes('settings')) {
      initSettingsFeatures();
    }

    // Feedback Page Features
    if (currentPage.includes('feedback')) {
      initFeedbackFeatures();
    }
  }

  // Profile page specific features
  function initProfileFeatures() {
    const editPreferencesBtn = document.querySelector('.edit-preferences-btn');

    if (editPreferencesBtn) {
      editPreferencesBtn.addEventListener('click', () => {
        // Redirect to settings page with preferences section focused
        localStorage.setItem('kuberai-settings-focus', 'preferences');
        window.location.href = '/settings';
      });
    }
  }

  // Settings page specific features
  function initSettingsFeatures() {
    // Handle form submissions
    const profileForm = document.getElementById('profile-form');
    const passwordForm = document.getElementById('password-form');
    const preferenceForm = document.getElementById('preference-form');
    const notificationForm = document.getElementById('notification-form');
    const dangerButtons = document.querySelectorAll('.danger-btn');

    if (profileForm) {
      profileForm.addEventListener('submit', (e) => {
        e.preventDefault();
        showSaveNotification('Profile information updated successfully');
      });
    }

    if (passwordForm) {
      passwordForm.addEventListener('submit', (e) => {
        e.preventDefault();

        // Basic password validation
        const newPassword = document.getElementById('new-password').value;
        const confirmPassword = document.getElementById('confirm-password').value;

        if (newPassword !== confirmPassword) {
          showError('New passwords do not match');
          return;
        }

        showSaveNotification('Password updated successfully');

        // Clear the form
        passwordForm.reset();
      });
    }

    if (preferenceForm) {
      preferenceForm.addEventListener('submit', (e) => {
        e.preventDefault();
        showSaveNotification('Investment preferences updated successfully');
      });
    }

    if (notificationForm) {
      notificationForm.addEventListener('submit', (e) => {
        e.preventDefault();
        showSaveNotification('Notification settings updated successfully');
      });
    }

    // Handle danger buttons
    if (dangerButtons.length > 0) {
      dangerButtons.forEach(btn => {
        btn.addEventListener('click', (e) => {
          const action = e.target.textContent.trim();

          if (action === 'Deactivate Account') {
            if (confirm('Are you sure you want to deactivate your account? You can reactivate it later.')) {
              showSaveNotification('Account deactivated');
              setTimeout(() => {
                window.location.href = '/login';
              }, 2000);
            }
          } else if (action === 'Delete Account') {
            if (confirm('WARNING: This will permanently delete your account and all your data. This action cannot be undone. Are you sure?')) {
              showSaveNotification('Account deleted');
              setTimeout(() => {
                window.location.href = '/login';
              }, 2000);
            }
          }
        });
      });
    }

    // Check if we need to scroll to a specific section
    const focusSection = localStorage.getItem('kuberai-settings-focus');
    if (focusSection) {
      setTimeout(() => {
        const section = document.querySelector(`.settings-section:nth-child(${focusSection === 'preferences' ? '3' : '1'})`);
        if (section) {
          section.scrollIntoView({ behavior: 'smooth' });
          section.classList.add('highlight');
          setTimeout(() => {
            section.classList.remove('highlight');
          }, 2000);
        }
        localStorage.removeItem('kuberai-settings-focus');
      }, 500);
    }
  }

  // Feedback page specific features
  function initFeedbackFeatures() {
    const feedbackForm = document.getElementById('feedback-form');
    const starRating = document.querySelectorAll('.star');
    let currentRating = 0;

    // Star rating functionality
    if (starRating.length > 0) {
      starRating.forEach(star => {
        // Hover effect
        star.addEventListener('mouseover', () => {
          const rating = parseInt(star.getAttribute('data-rating'));
          highlightStars(rating);
        });

        // Reset on mouseout if no rating selected
        star.addEventListener('mouseout', () => {
          if (currentRating === 0) {
            resetStars();
          } else {
            highlightStars(currentRating);
          }
        });

        // Set rating on click
        star.addEventListener('click', () => {
          currentRating = parseInt(star.getAttribute('data-rating'));
          highlightStars(currentRating);
        });
      });
    }

    // Form submission
    if (feedbackForm) {
      feedbackForm.addEventListener('submit', (e) => {
        e.preventDefault();

        // Get form values
        const category = document.getElementById('feedback-category').value;
        const feedbackText = document.getElementById('feedback-text').value;

        // Validate
        if (!category) {
          showError('Please select a feedback category');
          return;
        }

        if (!feedbackText.trim()) {
          showError('Please provide some feedback text');
          return;
        }

        if (currentRating === 0) {
          showError('Please provide a star rating');
          return;
        }

        // Show success message
        showSaveNotification('Thank you for your feedback!');

        // Reset the form
        feedbackForm.reset();
        currentRating = 0;
        resetStars();

        // Add to the previous feedback list
        addToPreviousFeedback(category, feedbackText);
      });
    }

    // Highlight stars function
    function highlightStars(rating) {
      starRating.forEach(star => {
        const starRating = parseInt(star.getAttribute('data-rating'));
        if (starRating <= rating) {
          star.classList.add('active');
        } else {
          star.classList.remove('active');
        }
      });
    }

    // Reset stars function
    function resetStars() {
      starRating.forEach(star => {
        star.classList.remove('active');
      });
    }

    // Add to previous feedback
    function addToPreviousFeedback(category, text) {
      const previousFeedback = document.querySelector('.previous-feedback');

      if (previousFeedback) {
        const today = new Date();
        const formattedDate = `${today.toLocaleString('default', { month: 'short' })} ${today.getDate()}, ${today.getFullYear()}`;

        const feedbackItem = document.createElement('div');
        feedbackItem.className = 'feedback-item';
        feedbackItem.innerHTML = `
          <div class="feedback-header">
            <div class="feedback-topic">${category}</div>
            <div class="feedback-date">${formattedDate}</div>
          </div>
          <div class="feedback-content">
            ${text}
          </div>
          <div class="feedback-status">Status: <span class="status-tag">Under Review</span></div>
        `;

        // Add to the top of the list
        previousFeedback.insertBefore(feedbackItem, previousFeedback.firstChild);
      }
    }
  }

  // Initialize slideable panels functionality
  function initSlidablePanels() {
    // Create sidebar toggle button if it doesn't exist
    if (!document.querySelector(".sidebar-toggle-btn")) {
      const sidebarToggleBtn = document.createElement("button");
      sidebarToggleBtn.className = "sidebar-toggle-btn";
      sidebarToggleBtn.innerHTML = '<i class="fas fa-bars"></i>';
      document.body.appendChild(sidebarToggleBtn);

      sidebarToggleBtn.addEventListener("click", function() {
        const sidebar = document.querySelector(".sidebar");
        if (sidebar) {
          sidebar.classList.toggle("collapsed");
        }
      });
    }

    // Create recommendation toggle button if it doesn't exist
    if (!document.querySelector(".recommendation-toggle")) {
      const recommendationToggleBtn = document.createElement("button");
      recommendationToggleBtn.className = "recommendation-toggle";
      recommendationToggleBtn.innerHTML = '<i class="fas fa-sliders-h"></i>';
      document.body.appendChild(recommendationToggleBtn);

      recommendationToggleBtn.addEventListener("click", function() {
        const recommendationArea = document.querySelector(".recommendation-area");
        if (recommendationArea) {
          recommendationArea.classList.toggle("collapsed");
        }
      });
    }

    // Initialize default state - collapsed on mobile
    const sidebar = document.querySelector(".sidebar");
    const recommendationArea = document.querySelector(".recommendation-area");

    if (window.innerWidth <= 768 && sidebar) {
      sidebar.classList.add("collapsed");
    }

    if (recommendationArea) {
      recommendationArea.classList.add("collapsed"); // Default collapsed
    }
  }

  // Set up all UI controls
  function setupUIControls() {
    // Only add chat controls if we're on chat page
    if (document.querySelector(".chat-area")) {
      // Add chat controls container if needed
      const chatControls = document.querySelector(".chat-controls") || document.createElement("div");
      if (!chatControls.classList.contains("chat-controls")) {
        chatControls.className = "chat-controls";
        document.querySelector(".chat-area").appendChild(chatControls);
      }

      // Add clear chat button
      if (!document.getElementById("clear-chat")) {
        const clearChatBtn = document.createElement("button");
        clearChatBtn.id = "clear-chat";
        clearChatBtn.className = "action-button";
        clearChatBtn.innerHTML = '<i class="fas fa-trash"></i> Clear';
        chatControls.appendChild(clearChatBtn);

        clearChatBtn.addEventListener("click", () => {
          if (confirm("Are you sure you want to clear the chat history?")) {
            chatBox.innerHTML = "";
            localStorage.removeItem("kuberai-chat-history");
          }
        });
      }

      // Add dark mode toggle
      if (!document.getElementById("dark-mode-toggle")) {
        const darkModeToggle = document.createElement("button");
        darkModeToggle.id = "dark-mode-toggle";
        darkModeToggle.className = "action-button";
        darkModeToggle.innerHTML = isDarkMode ?
          '<i class="fas fa-sun"></i> Light' :
          '<i class="fas fa-moon"></i> Dark';
        chatControls.appendChild(darkModeToggle);

        darkModeToggle.addEventListener("click", () => {
          isDarkMode = !isDarkMode;
          document.body.classList.toggle("dark-mode");
          localStorage.setItem("kuberai-dark-mode", isDarkMode);

          if (isDarkMode) {
            darkModeToggle.innerHTML = '<i class="fas fa-sun"></i> Light';
          } else {
            darkModeToggle.innerHTML = '<i class="fas fa-moon"></i> Dark';
          }
        });
      }

      // Add settings dropdown
      if (!document.querySelector(".settings-dropdown")) {
        const settingsDropdown = document.createElement("div");
        settingsDropdown.className = "settings-dropdown";

        const settingsBtn = document.createElement("button");
        settingsBtn.className = "settings-btn";
        settingsBtn.innerHTML = '<i class="fas fa-cog"></i>';

        const settingsMenu = document.createElement("div");
        settingsMenu.className = "settings-menu";

        // Add settings items
        const settingsItems = [
          { icon: "fas fa-download", text: "Export Chat" },
          { icon: "fas fa-language", text: "Language" },
          { icon: "fas fa-font", text: "Font Size" }
        ];

        settingsItems.forEach(item => {
          const settingsItem = document.createElement("div");
          settingsItem.className = "settings-item";
          settingsItem.innerHTML = `<i class="${item.icon}"></i> ${item.text}`;
          settingsMenu.appendChild(settingsItem);
        });

        settingsDropdown.appendChild(settingsBtn);
        settingsDropdown.appendChild(settingsMenu);
        chatControls.appendChild(settingsDropdown);

        settingsBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          settingsMenu.classList.toggle("active");
        });

        // Close settings when clicking outside
        document.addEventListener("click", (e) => {
          if (!settingsDropdown.contains(e.target)) {
            settingsMenu.classList.remove("active");
          }
        });
      }

      // Add quick actions section if needed
      let quickActionsContainer = document.querySelector(".quick-actions");
      if (!quickActionsContainer) {
        quickActionsContainer = document.createElement("div");
        quickActionsContainer.className = "quick-actions";
        quickActionsContainer.innerHTML = "<h4>Quick Actions</h4>";
        document.querySelector(".chat-area").appendChild(quickActionsContainer);

        // Add quick action buttons
        const quickActions = [
          "What is compound interest?",
          "How do ETFs work?",
          "Explain investment diversification",
          "What is dollar-cost averaging?"
        ];

        quickActions.forEach(action => {
          const actionBtn = document.createElement("button");
          actionBtn.className = "quick-action-btn";
          actionBtn.textContent = action;
          actionBtn.addEventListener("click", () => {
            chatInput.value = action;
            chatForm.dispatchEvent(new Event("submit"));
          });
          quickActionsContainer.appendChild(actionBtn);
        });
      }

      // Setup recommendation area if needed (only on chat page)
      if (!document.querySelector(".recommendation-area") && document.querySelector(".main")) {
        const recommendationArea = document.createElement("div");
        recommendationArea.className = "recommendation-area collapsed";
        recommendationArea.innerHTML = `
          <h3>Investment Preferences</h3>
          <div class="input-section">
            <div class="input-label">Risk Tolerance</div>
            <div class="input-group">
              <button class="pill-button">Conservative</button>
              <button class="pill-button selected">Moderate</button>
              <button class="pill-button">Aggressive</button>
            </div>

            <div class="input-label">Investment Horizon</div>
            <div class="slider-container">
              <input type="range" min="1" max="30" value="10" id="horizon-slider">
              <div class="slider-value">10 years</div>
            </div>
            <small class="horizon-hint">Drag to adjust your timeline</small>

            <div class="input-label">Investment Goals</div>
            <div class="input-group">
              <button class="pill-button">Retirement</button>
              <button class="pill-button selected">Growth</button>
              <button class="pill-button">Income</button>
              <button class="pill-button">Preservation</button>
            </div>

            <button class="recommend-btn">Update Preferences</button>
          </div>
        `;
        document.querySelector(".main").appendChild(recommendationArea);

        // Setup slider functionality
        const horizonSlider = recommendationArea.querySelector("#horizon-slider");
        const sliderValue = recommendationArea.querySelector(".slider-value");

        if (horizonSlider) {
          horizonSlider.addEventListener("input", () => {
            sliderValue.textContent = `${horizonSlider.value} years`;
          });
        }

        // Setup pill button selection
        const pillButtons = recommendationArea.querySelectorAll(".pill-button");
        pillButtons.forEach(button => {
          button.addEventListener("click", () => {
            // If in same group, deselect siblings
            const group = button.closest(".input-group");
            group.querySelectorAll(".pill-button").forEach(b => {
              b.classList.remove("selected");
            });
            button.classList.add("selected");
          });
        });

        // Setup update button
        const updateBtn = recommendationArea.querySelector(".recommend-btn");
        if (updateBtn) {
          updateBtn.addEventListener("click", () => {
            appendMessage("KuberAI", "Your investment preferences have been updated! I'll tailor my recommendations accordingly.", "bot");
            if (window.innerWidth <= 768) {
              recommendationArea.classList.add("collapsed");
            }
          });
        }
      }
    }
  }

  // Set up all event listeners
  function setupEventListeners() {
    // Chat form submit handler - only if we're on chat page
    if (chatForm) {
      chatForm.addEventListener("submit", async function (e) {
        e.preventDefault();

        const userMessage = chatInput.value.trim();
        if (userMessage === "") return;

        appendMessage("You", userMessage, "user");
        chatInput.value = "";

        // Show typing indicator
        const typingIndicatorId = showTypingIndicator();

        try {
          const response = await fetch("/api/ask", {
            method: "POST",
            headers: {
              "Content-Type": "application/json"
            },
            body: JSON.stringify({ question: userMessage })
          });

          // Remove typing indicator
          removeTypingIndicator(typingIndicatorId);

          const data = await response.json();
          if (data.status === "success") {
            appendMessage("KuberAI", data.answer, "bot");

            // Show relevant quick follow-ups based on the answer
            showRelevantFollowUps(data.answer);
          } else {
            appendMessage("KuberAI", `Error: ${data.message || "I couldn't understand that."}`, "bot");
          }
        } catch (error) {
          // Remove typing indicator
          removeTypingIndicator(typingIndicatorId);
          appendMessage("KuberAI", "Error reaching the backend. Please try again later.", "bot");
        }
      });
    }

    // File upload trigger
    if (uploadBtn && fileInput) {
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

        // Show typing indicator with progress
        const typingIndicatorId = showTypingIndicator(true);

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
            statusCheckInterval = setInterval(() => checkProcessingStatus(typingIndicatorId), 3000);
          } else {
            // Remove typing indicator
            removeTypingIndicator(typingIndicatorId);
            appendMessage("KuberAI", "Failed to process PDF. Please try again.", "bot");
          }
        } catch (err) {
          // Remove typing indicator
          removeTypingIndicator(typingIndicatorId);
          appendMessage("KuberAI", "Error uploading file. Please try again.", "bot");
        }
      });
    }

    // Enable send button when input has text (only if we're on chat page)
    if (chatInput) {
      chatInput.addEventListener("input", () => {
        const sendButton = document.querySelector("#chat-form button[type='submit']");
        if (sendButton) {
          sendButton.disabled = chatInput.value.trim() === "";
        }
      });
    }

    // Responsive behavior
    window.addEventListener('resize', () => {
      const sidebar = document.querySelector(".sidebar");
      const recommendationArea = document.querySelector(".recommendation-area");

      if (window.innerWidth <= 768) {
        if (sidebar) sidebar.classList.add("collapsed");
        if (recommendationArea) recommendationArea.classList.add("collapsed");
      }
    });
  }

  // Add emoji picker and voice input
  function setupEmojiAndVoiceInput() {
    // Add emoji picker if needed
    if (!document.getElementById("emoji-picker") && chatForm) {
      const emojiButton = document.createElement("button");
      emojiButton.id = "emoji-picker";
      emojiButton.className = "emoji-btn";
      emojiButton.innerHTML = "😊";
      emojiButton.type = "button";
      chatForm.insertBefore(emojiButton, chatForm.firstChild);

      // Simple emoji picker functionality
      emojiButton.addEventListener("click", () => {
        const emojis = ["😊", "👍", "🤔", "💰", "📈", "📉", "🏦", "💳", "💵", "🪙"];
        const pickerDiv = document.createElement("div");
        pickerDiv.className = "emoji-picker-dropdown";

        emojis.forEach(emoji => {
          const emojiSpan = document.createElement("span");
          emojiSpan.textContent = emoji;
          emojiSpan.addEventListener("click", () => {
            chatInput.value += emoji;
            pickerDiv.remove();
            chatInput.focus();
          });
          pickerDiv.appendChild(emojiSpan);
        });

        // Remove any existing picker
        const existingPicker = document.querySelector(".emoji-picker-dropdown");
        if (existingPicker) existingPicker.remove();

        // Add new picker
        emojiButton.parentNode.appendChild(pickerDiv);

        // Close picker when clicking outside
        document.addEventListener("click", function closeEmojiPicker(e) {
          if (!pickerDiv.contains(e.target) && e.target !== emojiButton) {
            pickerDiv.remove();
            document.removeEventListener("click", closeEmojiPicker);
          }
        });
      });
    }

    // Add speech recognition for voice input (if supported)
    if ('webkitSpeechRecognition' in window && !document.getElementById("voice-input") && chatForm) {
      const voiceButton = document.createElement("button");
      voiceButton.id = "voice-input";
      voiceButton.className = "voice-btn";
      voiceButton.innerHTML = '<i class="fas fa-microphone"></i>';
      voiceButton.type = "button";
      chatForm.insertBefore(voiceButton, chatInput);

      const recognition = new webkitSpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.lang = 'en-US';

      voiceButton.addEventListener("click", () => {
        if (voiceButton.classList.contains("listening")) {
          recognition.stop();
          voiceButton.classList.remove("listening");
          voiceButton.innerHTML = '<i class="fas fa-microphone"></i>';
        } else {
          recognition.start();
          voiceButton.classList.add("listening");
          voiceButton.innerHTML = '<i class="fas fa-microphone-slash"></i>';
        }
      });

      recognition.onresult = function(event) {
        const transcript = event.results[0][0].transcript;
        chatInput.value = transcript;
        voiceButton.classList.remove("listening");
        voiceButton.innerHTML = '<i class="fas fa-microphone"></i>';
      };

      recognition.onend = function() {
        voiceButton.classList.remove("listening");
        voiceButton.innerHTML = '<i class="fas fa-microphone"></i>';
      };
    }
  }

  // Load saved chat history from localStorage
  function loadChatHistory() {
    const savedHistory = localStorage.getItem("kuberai-chat-history");
    if (savedHistory) {
      const messages = JSON.parse(savedHistory);
      messages.forEach(msg => {
        appendMessage(msg.sender, msg.content, msg.type, false, msg.timestamp);
      });
    } else {
      // First time welcome message
      appendMessage("KuberAI", "👋 Welcome to KuberAI! I'm your financial assistant. Ask me about stocks, investing, or upload a financial document for analysis.", "bot");
    }
  }

  // Check processing status
  async function checkProcessingStatus(typingIndicatorId) {
    if (!currentTaskId) {
      clearInterval(statusCheckInterval);
      removeTypingIndicator(typingIndicatorId);
      return;
    }

    try {
      const response = await fetch(`/check_processing/${currentTaskId}`);
      const result = await response.json();

      // Update progress indicator if available
      if (result.progress) {
        updateProgress(typingIndicatorId, result.progress);
      }

      if (result.status === "complete") {
        clearInterval(statusCheckInterval);
        removeTypingIndicator(typingIndicatorId);
        appendMessage("KuberAI", "✅ PDF processed successfully! You can now ask me questions about the document.", "bot");

        // Add suggested questions about the document
        showDocumentQuestions();

        currentTaskId = null;
      } else if (result.status === "error") {
        clearInterval(statusCheckInterval);
        removeTypingIndicator(typingIndicatorId);
        appendMessage("KuberAI", `❌ ${result.message || "Error processing PDF."}`, "bot");
        currentTaskId = null;
      }
      // If still processing, continue waiting

    } catch (err) {
      console.error("Error checking status:", err);
      // Don't clear interval, try again
    }
  }

  // Generate and show relevant follow-up questions
  function showRelevantFollowUps(answer) {
    // Clear existing follow-ups
    const existingFollowUps = document.querySelector(".follow-ups");
    if (existingFollowUps) {
      existingFollowUps.remove();
    }

    // Simple keyword-based follow-up generation
    const followUps = [];

    if (answer.toLowerCase().includes("stock")) {
      followUps.push("What's the difference between stocks and bonds?");
    }

    if (answer.toLowerCase().includes("invest")) {
      followUps.push("What are some low-risk investment options?");
    }

    if (answer.toLowerCase().includes("portfolio")) {
      followUps.push("How often should I rebalance my portfolio?");
    }

    if (answer.toLowerCase().includes("tax")) {
      followUps.push("How can I reduce my investment tax liability?");
    }

    // Always include one general follow-up
    followUps.push("Can you explain dollar-cost averaging?");

    // Only show if we have follow-ups (max 3)
    if (followUps.length > 0) {
      const followUpsContainer = document.createElement("div");
      followUpsContainer.className = "follow-ups";
      followUpsContainer.innerHTML = "<p>Follow up questions:</p>";

      followUps.slice(0, 3).forEach(question => {
        const btn = document.createElement("button");
        btn.className = "follow-up-btn";
        btn.textContent = question;
        btn.addEventListener("click", () => {
          chatInput.value = question;
          chatForm.dispatchEvent(new Event("submit"));
        });
        followUpsContainer.appendChild(btn);
      });

      chatBox.appendChild(followUpsContainer);
      chatBox.scrollTop = chatBox.scrollHeight;
    }
  }
  
  // Show document-related questions
  function showDocumentQuestions() {
    const docQuestions = [
      "What are the key findings in this document?",
      "Summarize the main financial data in this document",
      "What are the financial metrics in this report?"
    ];
    
    const docQuestionsContainer = document.createElement("div");
    docQuestionsContainer.className = "document-questions";
    docQuestionsContainer.innerHTML = "<p>Ask about the document:</p>";
    
    docQuestions.forEach(question => {
      const btn = document.createElement("button");
      btn.className = "document-question-btn";
      btn.textContent = question;
      btn.addEventListener("click", () => {
        chatInput.value = question;
        chatForm.dispatchEvent(new Event("submit"));
      });
      docQuestionsContainer.appendChild(btn);
    });
    
    chatBox.appendChild(docQuestionsContainer);
    chatBox.scrollTop = chatBox.scrollHeight;
  }
  
  // Save message to history
  function saveMessage(sender, content, type, timestamp) {
    const savedHistory = localStorage.getItem("kuberai-chat-history") || "[]";
    const messages = JSON.parse(savedHistory);
    messages.push({ 
      sender, 
      content, 
      type, 
      timestamp: timestamp || new Date().toISOString() 
    });
    
    // Keep history limited to last 50 messages to prevent localStorage overflow
    if (messages.length > 50) {
      messages.shift();
    }
    
    localStorage.setItem("kuberai-chat-history", JSON.stringify(messages));
  }
  
  // Helper functions for typing indicators
  function showTypingIndicator(withProgress = false) {
    const indicatorId = `typing-${Date.now()}`;
    const indicatorElement = document.createElement("div");
    indicatorElement.id = indicatorId;
    indicatorElement.className = "message bot typing-indicator";
    
    let innerHTML = `
      <div class="typing-dots">
        <span class="dot"></span>
        <span class="dot"></span>
        <span class="dot"></span>
      </div>
    `;
    
    if (withProgress) {
      innerHTML += `
        <div class="progress-container">
          <div class="progress-bar">
            <div class="progress-bar-fill" style="width: 0%"></div>
          </div>
          <div class="progress-text">0% complete</div>
        </div>
      `;
    }
    
    indicatorElement.innerHTML = innerHTML;
    chatBox.appendChild(indicatorElement);
    chatBox.scrollTop = chatBox.scrollHeight;
    return indicatorId;
  }
  
  function removeTypingIndicator(indicatorId) {
    const indicatorElement = document.getElementById(indicatorId);
    if (indicatorElement) {
      indicatorElement.remove();
    }
  }
  
  // Update progress in typing indicator
  function updateProgress(indicatorId, progress) {
    const indicator = document.getElementById(indicatorId);
    if (indicator && indicator.querySelector(".progress-bar")) {
      const progressBar = indicator.querySelector(".progress-bar-fill");
      progressBar.style.width = `${progress}%`;
      indicator.querySelector(".progress-text").textContent = `${progress}% complete`;
    }
  }
  
  function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }
  
  // Append message to chat
  function appendMessage(sender, message, type, save = true, timestamp = null) {
    const now = timestamp || new Date().toISOString();
    const bubble = document.createElement("div");
    bubble.className = `message ${type}`;
    
    if (type === "user") {
      bubble.innerHTML = `
        <div class="message-content">
          <div class="message-header">
            <strong>${sender}</strong>
            <span class="timestamp">${formatTimestamp(now)}</span>
          </div>
          <div class="message-body">${message}</div>
        </div>
      `;
    } else {
      // Use marked library if available, otherwise use message directly
      const parsedMessage = window.marked ? marked.parse(message) : message;
      bubble.innerHTML = `
        <div class="message-content">
          <div class="message-header">
            <strong>${sender}</strong>
            <span class="timestamp">${formatTimestamp(now)}</span>
          </div>
          <div class="message-body">${parsedMessage}</div>
        </div>
      `;
      
      // Add feedback buttons
      addFeedbackButtons(bubble);
    }
    
    chatBox.appendChild(bubble);
    chatBox.scrollTop = chatBox.scrollHeight;
    
    // Save to history if needed
    if (save) {
      saveMessage(sender, message, type, now);
    }
  }
  
  // Add feedback buttons for bot responses
  function addFeedbackButtons(messageElement) {
    const feedbackDiv = document.createElement("div");
    feedbackDiv.className = "feedback-buttons";
    
    const thumbsUp = document.createElement("button");
    thumbsUp.innerHTML = '<i class="fas fa-thumbs-up"></i>';
    thumbsUp.className = "feedback-btn";
    thumbsUp.addEventListener("click", () => {
      // Here you would send positive feedback to your backend
      feedbackDiv.innerHTML = "<span class='feedback-thanks'>Thanks for your feedback!</span>";
      setTimeout(() => {
        feedbackDiv.remove();
      }, 2000);
    });
    
    const thumbsDown = document.createElement("button");
    thumbsDown.innerHTML = '<i class="fas fa-thumbs-down"></i>';
    thumbsDown.className = "feedback-btn";
    thumbsDown.addEventListener("click", () => {
      // Here you would send negative feedback to your backend
      feedbackDiv.innerHTML = "<span class='feedback-thanks'>Thanks for your feedback!</span>";
      setTimeout(() => {
        feedbackDiv.remove();
      }, 2000);
    });
    
    feedbackDiv.appendChild(thumbsUp);
    feedbackDiv.appendChild(thumbsDown);
    messageElement.appendChild(feedbackDiv);
  }
});