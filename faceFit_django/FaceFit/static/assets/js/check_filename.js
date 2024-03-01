// static/js/check_filename.js
// window.addEventListener('DOMContentLoaded', function() {
//     const currentURL = window.location.href;
//
//     // Check if the URL matches the desired patterns
//     if (currentURL.includes('/admin/FaceFit/reference/') && (currentURL.endsWith('/add/') || currentURL.includes('/change/'))) {
//         // Your existing JavaScript code here
//         const referenceTitleInput = document.getElementById('id_reference_title');
//         const saveButton = document.querySelector('.submit-row input[name="_save"]');
//         const saveAndAdd = document.querySelector('.submit-row input[name="_addanother"]');
//         const saveAndContinue = document.querySelector('.submit-row input[name="_continue"]');
//
//         let errorMessageSpan = referenceTitleInput.parentElement.querySelector('.error-message');
//
//         if (!errorMessageSpan) {
//             errorMessageSpan = document.createElement('span');
//             errorMessageSpan.className = 'error-message';
//             referenceTitleInput.parentElement.appendChild(errorMessageSpan);
//         }
//
//         if (referenceTitleInput) {
//             referenceTitleInput.addEventListener('input', function () {
//                 let referenceTitle = this.value.trim();
//                 const folderPath = '/static/assets/images/';
//                 const possibleExtensions = ['.jpg'];
//
//
//                 errorMessageSpan.textContent = '';
//
//                 // Try to find the original file or different extensions
//                 let matchingFile = null;
//                 for (let i = 0; i < possibleExtensions.length; i++) {
//                     const ext = possibleExtensions[i];
//                     const filePath = folderPath + referenceTitle + ext;
//
//                     if (fileExists(filePath)) {
//                         matchingFile = filePath;
//                         referenceTitle += ext;
//                         console.log(referenceTitle, 'found');
//                         break;
//                     }
//                 }
//
//                 // Display error message if no matching file is found
//                 if (!matchingFile) {
//                     errorMessageSpan.textContent = 'File not found';
//                 } else {
//                     // Update the input value with the adjusted title
//                     referenceTitleInput.value = referenceTitle;
//                     console.log('File found:', matchingFile);
//                 }
//                 // Disable save button if no matching file is found
//                 saveButton.disabled = !matchingFile;
//                 saveAndAdd.disabled = !matchingFile;
//                 saveAndContinue.disabled = !matchingFile;
//
//                 // You can do further actions based on the matchingFile, such as displaying a message to the user.
//                 if (matchingFile) {
//                     console.log('File found:', matchingFile);
//
//                 } else {
//                     console.log('No matching file found.');
//                 }
//             });
//         }
//
//         function fileExists(url) {
//             var http = new XMLHttpRequest();
//             http.open('HEAD', url, false);
//             http.send();
//             return http.status !== 404;
//         }
//     }
// });
window.addEventListener('DOMContentLoaded', function() {
    const referenceTitleInput = document.getElementById('id_reference_title');
    const sourceSelect = document.getElementById('id_source');

//     if (referenceTitleInput && sourceSelect) {
//         sourceSelect.addEventListener('change', function() {
//             // Get the selected option
//             if (this.files.length > 0) {
//                 // Get the selected file name
//
//                 console.log('file name', this.files[0].name)
//                 // Update the reference title input
//                 referenceTitleInput.value = this.files[0].name;
//
//             }
//             // Your existing logic here, if needed
//         });
//     }
});
// Add this to your check_filename.js
// window.addEventListener('DOMContentLoaded', function() {
//     const referenceTitleInput = document.getElementById('id_reference_title');
//     const sourceFileInput = document.getElementById('id_source');
//     const customFileInputLabel = document.querySelector('.custom-file-input-label');
//
//     if (referenceTitleInput && sourceFileInput && customFileInputLabel) {
//         customFileInputLabel.addEventListener('click', function() {
//             // Trigger a click on the hidden file input
//             sourceFileInput.click();
//         });
//
//         sourceFileInput.addEventListener('change', function() {
//             // Check if a file is selected
//             if (this.files.length > 0) {
//                 // Get the selected file name
//                 const fileName = this.files[0].name;
//
//                 // Update the reference title input
//                 referenceTitleInput.value = fileName;
//
//                 // Your existing logic here, if needed
//             }
//         });
//     }
// });