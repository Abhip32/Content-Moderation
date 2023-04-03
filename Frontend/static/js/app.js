const imageUpload = document.querySelector('#image');
imageUpload.addEventListener('change', (event) => { // Get the selected file
    const selectedFile = event.target.files[0];

    // Create a FileReader object
    const reader = new FileReader();

    // Set the onload handler for the FileReader object
    reader.onload = (event) => { // Get the data URL
        const dataUrl = event.target.result;

        // Create an <img> element with the data URL as the src attribute
        const img = document.getElementById('output')
        img.setAttribute('src', dataUrl);

    };

    // Read the selected file as a data URL
    reader.readAsDataURL(selectedFile);
});