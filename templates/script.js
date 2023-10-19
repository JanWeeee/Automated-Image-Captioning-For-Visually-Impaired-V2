document.getElementById('capture-button').addEventListener('click', function() {
    var uploadForm = document.getElementById('upload-form');
    var captureButton = document.getElementById('capture-button');
    var imageInput = document.getElementById('image-input');
    var imageContainer = document.getElementById('image-container');
    var captureImageBtn = document.createElement('button');

document.getElementById('image-input').addEventListener('change', function(event) {
    var uploadedImage = document.getElementById('uploaded-image');
    uploadedImage.src = URL.createObjectURL(event.target.files[0]);
    document.getElementById('image-container').style.display = 'block';
});

    // Access the user's camera
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(function(stream) {
            // Create a video element to display the camera feed
            var video = document.createElement('video');
            video.setAttribute('autoplay', 'true');
            video.srcObject = stream;
            document.body.appendChild(video);

            // Create a canvas element to capture the image from the camera feed
            var canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            var context = canvas.getContext('2d');

            captureImageBtn.textContent = 'Capture';
            captureImageBtn.addEventListener('click', function() {
                // Draw the current frame from the camera onto the canvas
                context.drawImage(video, 0, 0, canvas.width, canvas.height);
                
                // Create an image element to display the captured image
                var capturedImage = new Image();
                capturedImage.src = canvas.toDataURL('image/png');
                imageContainer.innerHTML = ''; // Clear previous content
                imageContainer.appendChild(capturedImage);

                // Show the options to upload or discard the captured image
                var uploadButton = document.createElement('button');
                uploadButton.textContent = 'Upload';
                uploadButton.addEventListener('click', function() {
                    // Set the captured image as the file for upload
                    var capturedImageDataUrl = capturedImage.src;
                    var capturedImageBlob = dataURItoBlob(capturedImageDataUrl);
                    var capturedImageFile = new File([capturedImageBlob], 'captured-image.png');
                    imageInput.files = new FileList([capturedImageFile]);

                    // Display the uploaded image
                    var uploadedImage = document.getElementById('uploaded-image');
                    uploadedImage.src = capturedImageDataUrl;

                    // Hide camera elements
                    video.style.display = 'none';
                    canvas.style.display = 'none';
                    captureImageBtn.style.display = 'none';
                    uploadButton.style.display = 'none';

                    // Show upload form and buttons
                    uploadForm.style.display = 'block';
                    captureButton.style.display = 'block';
                });

                var discardButton = document.createElement('button');
                discardButton.textContent = 'Discard';
                discardButton.addEventListener('click', function() {
                    // Hide camera elements
                    video.style.display = 'none';
                    canvas.style.display = 'none';
                    captureImageBtn.style.display = 'none';
                    uploadButton.style.display = 'none';
                    
                    // Show upload form and buttons
                    uploadForm.style.display = 'block';
                    captureButton.style.display = 'block';
                });

                imageContainer.appendChild(uploadButton);
                imageContainer.appendChild(discardButton);
            });

            document.body.appendChild(captureImageBtn);
        })
        .catch(function(error) {
            console.error('Error accessing the camera: ', error);
        });
});

// Function to convert data URI to Blob
function dataURItoBlob(dataURI) {
    var byteString = atob(dataURI.split(',')[1]);
    var ab = new ArrayBuffer(byteString.length);
    var ia = new Uint8Array(ab);
    for (var i = 0; i < byteString.length; i++) {
        ia[i] = byteString.charCodeAt(i);
    }
    return new Blob([ab], { type: 'image/png' });
}
