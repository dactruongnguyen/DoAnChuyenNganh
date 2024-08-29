// add hovered class to selected list item
let list = document.querySelectorAll(".navigation li");

function activeLink() {
  list.forEach((item) => {
    item.classList.remove("hovered");
  });
  this.classList.add("hovered");
}

list.forEach((item) => item.addEventListener("mouseover", activeLink));

// Menu Toggle
let toggle = document.querySelector(".toggle");
let navigation = document.querySelector(".navigation");
let main = document.querySelector(".main");

toggle.onclick = function () {
  navigation.classList.toggle("active");
  main.classList.toggle("active");
};

document
  .querySelector("#upload-form")
  .addEventListener("submit", function (event) {
    event.preventDefault(); // Ngăn chặn hành động mặc định của form
    var formData = new FormData(this);

    // Thêm mô hình đã chọn vào FormData
    var modelSelect = document.querySelector("#model-select");
    formData.append("model", modelSelect.value);

    fetch("/predict", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.disease) {
          document.getElementById("disease").innerText =
            "Dự đoán của mô hình : " + data.disease;
          // +
          // " (Tỉ lệ chính xác : " +
          // data.confidence +
          // "%)";
        } else if (data.error) {
          document.getElementById("disease").innerText = "Error: " + data.error;
        }
      })
      .catch((error) => {
        document.getElementById("disease").innerText = "Error: " + error;
      });
  });

document
  .getElementById("file-input")
  .addEventListener("change", function (event) {
    const file = event.target.files[0];
    const reader = new FileReader();

    reader.onload = function (e) {
      const img = document.getElementById("preview-image");
      img.src = e.target.result;
      img.style.display = "block";
    };

    if (file) {
      reader.readAsDataURL(file);
    }
  });
