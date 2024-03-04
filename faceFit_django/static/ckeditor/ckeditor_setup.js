// ckeditor_setup.js
document.addEventListener('DOMContentLoaded', function () {
    ClassicEditor
        .create(document.querySelector('#id_reference_text'), {
            language: 'en',
            toolbar: [
                'bold',
                'italic',
                '|',
                'link',
                '|',
                'undo',
                'redo'
            ],
            shouldNotGroupWhenFull: true
        })
        .catch(error => {
            console.error(error);
        });
});