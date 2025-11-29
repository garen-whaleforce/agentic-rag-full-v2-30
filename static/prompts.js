async function loadPrompts() {
  const resp = await fetch("/api/prompts");
  if (!resp.ok) {
    alert("載入 prompts 失敗：" + (await resp.text()));
    return;
  }

  const data = await resp.json();
  const container = document.getElementById("prompt-list");
  container.innerHTML = "";

  data.forEach((item) => {
    const wrapper = document.createElement("div");
    wrapper.className = "prompt-block";

    const label = document.createElement("label");
    label.textContent = item.key;
    label.style.display = "block";
    label.style.fontWeight = "bold";
    label.style.marginBottom = "4px";

    const hint = document.createElement("div");
    hint.className = "prompt-warning";
    hint.textContent = "提示：請盡量保留整體結構，只微調敘述與語氣。";

    const textarea = document.createElement("textarea");
    textarea.value = item.content;
    textarea.rows = 16;
    textarea.style.width = "100%";
    textarea.dataset.key = item.key;

    const saveBtn = document.createElement("button");
    saveBtn.textContent = "儲存這個 Prompt";
    saveBtn.style.marginTop = "8px";

    saveBtn.onclick = async () => {
      const payload = {
        key: item.key,
        content: textarea.value,
      };
      const res = await fetch("/api/prompts", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        alert("儲存失敗：" + (await res.text()));
      } else {
        alert("已儲存！");
      }
    };

    wrapper.appendChild(label);
    wrapper.appendChild(hint);
    wrapper.appendChild(textarea);
    wrapper.appendChild(saveBtn);

    container.appendChild(wrapper);
  });
}

document.getElementById("back-btn").onclick = () => {
  window.location.href = "/";
};

loadPrompts().catch((err) => {
  console.error("loadPrompts error", err);
  alert("載入 prompts 發生錯誤，請查看 console");
});
