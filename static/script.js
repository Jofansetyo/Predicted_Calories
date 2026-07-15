// Switch antar tab (Biodata / Aktivitas)
const tabBtns = document.querySelectorAll(".tab-btn");
const panels = document.querySelectorAll(".tab-panel");

function switchTab(tabName) {
  tabBtns.forEach((btn) => btn.classList.toggle("active", btn.dataset.tab === tabName));
  panels.forEach((panel) => panel.classList.toggle("active", panel.dataset.panel === tabName));
}

tabBtns.forEach((btn) => {
  btn.addEventListener("click", () => switchTab(btn.dataset.tab));
});

document.querySelectorAll(".next-btn").forEach((btn) => {
  btn.addEventListener("click", () => switchTab(btn.dataset.next));
});

document.getElementById("calorieForm").addEventListener("submit", async function (e) {
  e.preventDefault();

  const resultBox = document.getElementById("resultBox");
  const errorBox = document.getElementById("errorBox");
  resultBox.classList.add("hidden");
  errorBox.classList.add("hidden");

  const payload = {
    gender: document.getElementById("gender").value,
    age: document.getElementById("age").value,
    height: document.getElementById("height").value,
    weight: document.getElementById("weight").value,
    duration: document.getElementById("duration").value,
    heart_rate: document.getElementById("heart_rate").value,
    body_temp: document.getElementById("body_temp").value,
  };

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await res.json();

    if (!res.ok) {
      errorBox.textContent = data.error || "Terjadi kesalahan.";
      errorBox.classList.remove("hidden");
      return;
    }

    document.getElementById("resultValue").textContent = data.calories;
    document.getElementById("bmiBadge").textContent = "BMI " + data.bmi;
    document.getElementById("bmiLabel").textContent = data.label_bmi;
    document.getElementById("saranText").textContent = data.saran;
    resultBox.classList.remove("hidden");
  } catch (err) {
    errorBox.textContent = "Gagal terhubung ke server.";
    errorBox.classList.remove("hidden");
  }
});
