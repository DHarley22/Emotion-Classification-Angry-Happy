document.addEventListener('DOMContentLoaded', function() {
  const input = document.getElementById('image');
  const imageDisplay = document.getElementById('imageDisplay');

  input.addEventListener('change', function() {
    const file = input.files[0];
    const reader = new FileReader();

    reader.onload = function(e) {
      imageDisplay.src = e.target.result;
      imageDisplay.width = 1000;
      imageDisplay.height = 1000;
    }

    if (file) {
      reader.readAsDataURL(file);
    }
  });
});
