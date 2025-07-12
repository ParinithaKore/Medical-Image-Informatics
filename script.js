document.getElementById('uploadForm').addEventListener('submit', function(event) {
    event.preventDefault();

    let formData = new FormData();
    let files = document.getElementById('images').files;

    for (let i = 0; i < files.length; i++) {
        formData.append('images[]', files[i]);
    }

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        console.log(data);
        alert('Images uploaded successfully!');
    })
    .catch(error => {
        console.error(error);
        alert('An error occurred while uploading images.');
    });
});
