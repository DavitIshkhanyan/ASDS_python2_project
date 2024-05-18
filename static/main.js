const submitButton = document.getElementById('submitButton');
const fileInput = document.getElementById('imageInput');

submitButton.addEventListener('click', async function(e) {
    const file = fileInput.files[0];
    if (!file) {
        alert('Please select an image.');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/segment-image/', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        // console.log(data);

        const img = document.createElement('img');
        img.src = 'data:image/png;base64,' + data.mask_image;
        document.getElementById('segmentedImage').innerHTML = '';
        document.getElementById('segmentedImage').appendChild(img);
    } catch (error) {
        console.error('Error:', error);
    }
});