/* VIEW AND CAMERA */
const ref_img = document.getElementById("ref_img");
const video = document.getElementById("webcam");

/* HINTS */
const hints = document.getElementById("hints_section");
const percent_x = document.getElementById("pb_x");
const percent_y = document.getElementById("pb_y");
const percent_z = document.getElementById("pb_z");

/* CONTAINERS */
const body = document.body
const container = document.getElementsByClassName("container")[0]
const container_left = document.getElementById("ref_btns");
const container_right = document.getElementById("morph_btns")
const container_center = document.getElementById("main_view")

/* CANVAS */
const canvas = document.getElementById("canvas");
const cam_canvas = document.getElementById('cam_canvas');
const ctx = cam_canvas.getContext('2d');
const context = canvas.getContext("2d");

/* POPUP */
const popupButton = document.getElementById('popup-button');
const resetButton = document.getElementById('reset-button');
const popupWindow = document.getElementById('popup-window');
const emailInput = document.getElementById('email-input');
const sendEmailButton = document.getElementById('send-email-button');

const port = window.location.port;
const host = window.location.hostname;
const protocol = window.location.protocol;

let cam_face, detector, selected, detect_interval, m_all_btns;
let face_arr = []
let morphed = ''
const model = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;
const detectorConfig = {
    runtime: 'mediapipe',
    solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh',
    min_tracking_confidence: 0.2
}
// const morphs_path = '../morphs'
const default_view = '../images/Thumbs/default_view.jpg'
const default_morph = '../images/Thumbs/morph_thumb.jpg'
const send_logo = '../images/Thumbs/send.png'
const reset_logo = '../images/Thumbs/reset.png'
const start_view = '../images/Thumbs/start_view.jpg'

/* UI FUNCTIONS*/
const leftSlick = $("#ref_btns");
const rightSlick = $("#morph_btns")

const TRIANGULATION = [
    127, 34, 139, 11, 0, 37, 232, 231, 120, 72, 37, 39, 128, 121, 47, 232, 121,
    128, 104, 69, 67, 175, 171, 148, 157, 154, 155, 118, 50, 101, 73, 39, 40, 9,
    151, 108, 48, 115, 131, 194, 204, 211, 74, 40, 185, 80, 42, 183, 40, 92, 186,
    230, 229, 118, 202, 212, 214, 83, 18, 17, 76, 61, 146, 160, 29, 30, 56, 157,
    173, 106, 204, 194, 135, 214, 192, 203, 165, 98, 21, 71, 68, 51, 45, 4, 144,
    24, 23, 77, 146, 91, 205, 50, 187, 201, 200, 18, 91, 106, 182, 90, 91, 181,
    85, 84, 17, 206, 203, 36, 148, 171, 140, 92, 40, 39, 193, 189, 244, 159, 158,
    28, 247, 246, 161, 236, 3, 196, 54, 68, 104, 193, 168, 8, 117, 228, 31, 189,
    193, 55, 98, 97, 99, 126, 47, 100, 166, 79, 218, 155, 154, 26, 209, 49, 131,
    135, 136, 150, 47, 126, 217, 223, 52, 53, 45, 51, 134, 211, 170, 140, 67, 69,
    108, 43, 106, 91, 230, 119, 120, 226, 130, 247, 63, 53, 52, 238, 20, 242, 46,
    70, 156, 78, 62, 96, 46, 53, 63, 143, 34, 227, 173, 155, 133, 123, 117, 111,
    44, 125, 19, 236, 134, 51, 216, 206, 205, 154, 153, 22, 39, 37, 167, 200, 201,
    208, 36, 142, 100, 57, 212, 202, 20, 60, 99, 28, 158, 157, 35, 226, 113, 160,
    159, 27, 204, 202, 210, 113, 225, 46, 43, 202, 204, 62, 76, 77, 137, 123, 116,
    41, 38, 72, 203, 129, 142, 64, 98, 240, 49, 102, 64, 41, 73, 74, 212, 216,
    207, 42, 74, 184, 169, 170, 211, 170, 149, 176, 105, 66, 69, 122, 6, 168, 123,
    147, 187, 96, 77, 90, 65, 55, 107, 89, 90, 180, 101, 100, 120, 63, 105, 104,
    93, 137, 227, 15, 86, 85, 129, 102, 49, 14, 87, 86, 55, 8, 9, 100, 47, 121,
    145, 23, 22, 88, 89, 179, 6, 122, 196, 88, 95, 96, 138, 172, 136, 215, 58,
    172, 115, 48, 219, 42, 80, 81, 195, 3, 51, 43, 146, 61, 171, 175, 199, 81, 82,
    38, 53, 46, 225, 144, 163, 110, 246, 33, 7, 52, 65, 66, 229, 228, 117, 34,
    127, 234, 107, 108, 69, 109, 108, 151, 48, 64, 235, 62, 78, 191, 129, 209,
    126, 111, 35, 143, 163, 161, 246, 117, 123, 50, 222, 65, 52, 19, 125, 141,
    221, 55, 65, 3, 195, 197, 25, 7, 33, 220, 237, 44, 70, 71, 139, 122, 193, 245,
    247, 130, 33, 71, 21, 162, 153, 158, 159, 170, 169, 150, 188, 174, 196, 216,
    186, 92, 144, 160, 161, 2, 97, 167, 141, 125, 241, 164, 167, 37, 72, 38, 12,
    145, 159, 160, 38, 82, 13, 63, 68, 71, 226, 35, 111, 158, 153, 154, 101, 50,
    205, 206, 92, 165, 209, 198, 217, 165, 167, 97, 220, 115, 218, 133, 112, 243,
    239, 238, 241, 214, 135, 169, 190, 173, 133, 171, 208, 32, 125, 44, 237, 86,
    87, 178, 85, 86, 179, 84, 85, 180, 83, 84, 181, 201, 83, 182, 137, 93, 132,
    76, 62, 183, 61, 76, 184, 57, 61, 185, 212, 57, 186, 214, 207, 187, 34, 143,
    156, 79, 239, 237, 123, 137, 177, 44, 1, 4, 201, 194, 32, 64, 102, 129, 213,
    215, 138, 59, 166, 219, 242, 99, 97, 2, 94, 141, 75, 59, 235, 24, 110, 228,
    25, 130, 226, 23, 24, 229, 22, 23, 230, 26, 22, 231, 112, 26, 232, 189, 190,
    243, 221, 56, 190, 28, 56, 221, 27, 28, 222, 29, 27, 223, 30, 29, 224, 247,
    30, 225, 238, 79, 20, 166, 59, 75, 60, 75, 240, 147, 177, 215, 20, 79, 166,
    187, 147, 213, 112, 233, 244, 233, 128, 245, 128, 114, 188, 114, 217, 174,
    131, 115, 220, 217, 198, 236, 198, 131, 134, 177, 132, 58, 143, 35, 124, 110,
    163, 7, 228, 110, 25, 356, 389, 368, 11, 302, 267, 452, 350, 349, 302, 303,
    269, 357, 343, 277, 452, 453, 357, 333, 332, 297, 175, 152, 377, 384, 398,
    382, 347, 348, 330, 303, 304, 270, 9, 336, 337, 278, 279, 360, 418, 262, 431,
    304, 408, 409, 310, 415, 407, 270, 409, 410, 450, 348, 347, 422, 430, 434,
    313, 314, 17, 306, 307, 375, 387, 388, 260, 286, 414, 398, 335, 406, 418, 364,
    367, 416, 423, 358, 327, 251, 284, 298, 281, 5, 4, 373, 374, 253, 307, 320,
    321, 425, 427, 411, 421, 313, 18, 321, 405, 406, 320, 404, 405, 315, 16, 17,
    426, 425, 266, 377, 400, 369, 322, 391, 269, 417, 465, 464, 386, 257, 258,
    466, 260, 388, 456, 399, 419, 284, 332, 333, 417, 285, 8, 346, 340, 261, 413,
    441, 285, 327, 460, 328, 355, 371, 329, 392, 439, 438, 382, 341, 256, 429,
    420, 360, 364, 394, 379, 277, 343, 437, 443, 444, 283, 275, 440, 363, 431,
    262, 369, 297, 338, 337, 273, 375, 321, 450, 451, 349, 446, 342, 467, 293,
    334, 282, 458, 461, 462, 276, 353, 383, 308, 324, 325, 276, 300, 293, 372,
    345, 447, 382, 398, 362, 352, 345, 340, 274, 1, 19, 456, 248, 281, 436, 427,
    425, 381, 256, 252, 269, 391, 393, 200, 199, 428, 266, 330, 329, 287, 273,
    422, 250, 462, 328, 258, 286, 384, 265, 353, 342, 387, 259, 257, 424, 431,
    430, 342, 353, 276, 273, 335, 424, 292, 325, 307, 366, 447, 345, 271, 303,
    302, 423, 266, 371, 294, 455, 460, 279, 278, 294, 271, 272, 304, 432, 434,
    427, 272, 407, 408, 394, 430, 431, 395, 369, 400, 334, 333, 299, 351, 417,
    168, 352, 280, 411, 325, 319, 320, 295, 296, 336, 319, 403, 404, 330, 348,
    349, 293, 298, 333, 323, 454, 447, 15, 16, 315, 358, 429, 279, 14, 15, 316,
    285, 336, 9, 329, 349, 350, 374, 380, 252, 318, 402, 403, 6, 197, 419, 318,
    319, 325, 367, 364, 365, 435, 367, 397, 344, 438, 439, 272, 271, 311, 195, 5,
    281, 273, 287, 291, 396, 428, 199, 311, 271, 268, 283, 444, 445, 373, 254,
    339, 263, 466, 249, 282, 334, 296, 449, 347, 346, 264, 447, 454, 336, 296,
    299, 338, 10, 151, 278, 439, 455, 292, 407, 415, 358, 371, 355, 340, 345, 372,
    390, 249, 466, 346, 347, 280, 442, 443, 282, 19, 94, 370, 441, 442, 295, 248,
    419, 197, 263, 255, 359, 440, 275, 274, 300, 383, 368, 351, 412, 465, 263,
    467, 466, 301, 368, 389, 380, 374, 386, 395, 378, 379, 412, 351, 419, 436,
    426, 322, 373, 390, 388, 2, 164, 393, 370, 462, 461, 164, 0, 267, 302, 11, 12,
    374, 373, 387, 268, 12, 13, 293, 300, 301, 446, 261, 340, 385, 384, 381, 330,
    266, 425, 426, 423, 391, 429, 355, 437, 391, 327, 326, 440, 457, 438, 341,
    382, 362, 459, 457, 461, 434, 430, 394, 414, 463, 362, 396, 369, 262, 354,
    461, 457, 316, 403, 402, 315, 404, 403, 314, 405, 404, 313, 406, 405, 421,
    418, 406, 366, 401, 361, 306, 408, 407, 291, 409, 408, 287, 410, 409, 432,
    436, 410, 434, 416, 411, 264, 368, 383, 309, 438, 457, 352, 376, 401, 274,
    275, 4, 421, 428, 262, 294, 327, 358, 433, 416, 367, 289, 455, 439, 462, 370,
    326, 2, 326, 370, 305, 460, 455, 254, 449, 448, 255, 261, 446, 253, 450, 449,
    252, 451, 450, 256, 452, 451, 341, 453, 452, 413, 464, 463, 441, 413, 414,
    258, 442, 441, 257, 443, 442, 259, 444, 443, 260, 445, 444, 467, 342, 445,
    459, 458, 250, 289, 392, 290, 290, 328, 460, 376, 433, 435, 250, 290, 392,
    411, 416, 433, 341, 463, 464, 453, 464, 465, 357, 465, 412, 343, 412, 399,
    360, 363, 440, 437, 399, 456, 420, 456, 363, 401, 435, 288, 372, 383, 353,
    339, 255, 249, 448, 261, 255, 133, 243, 190, 133, 155, 112, 33, 246, 247, 33,
    130, 25, 398, 384, 286, 362, 398, 414, 362, 463, 341, 263, 359, 467, 263, 249,
    255, 466, 467, 260, 75, 60, 166, 238, 239, 79, 162, 127, 139, 72, 11, 37, 121,
    232, 120, 73, 72, 39, 114, 128, 47, 233, 232, 128, 103, 104, 67, 152, 175,
    148, 173, 157, 155, 119, 118, 101, 74, 73, 40, 107, 9, 108, 49, 48, 131, 32,
    194, 211, 184, 74, 185, 191, 80, 183, 185, 40, 186, 119, 230, 118, 210, 202,
    214, 84, 83, 17, 77, 76, 146, 161, 160, 30, 190, 56, 173, 182, 106, 194, 138,
    135, 192, 129, 203, 98, 54, 21, 68, 5, 51, 4, 145, 144, 23, 90, 77, 91, 207,
    205, 187, 83, 201, 18, 181, 91, 182, 180, 90, 181, 16, 85, 17, 205, 206, 36,
    176, 148, 140, 165, 92, 39, 245, 193, 244, 27, 159, 28, 30, 247, 161, 174,
    236, 196, 103, 54, 104, 55, 193, 8, 111, 117, 31, 221, 189, 55, 240, 98, 99,
    142, 126, 100, 219, 166, 218, 112, 155, 26, 198, 209, 131, 169, 135, 150, 114,
    47, 217, 224, 223, 53, 220, 45, 134, 32, 211, 140, 109, 67, 108, 146, 43, 91,
    231, 230, 120, 113, 226, 247, 105, 63, 52, 241, 238, 242, 124, 46, 156, 95,
    78, 96, 70, 46, 63, 116, 143, 227, 116, 123, 111, 1, 44, 19, 3, 236, 51, 207,
    216, 205, 26, 154, 22, 165, 39, 167, 199, 200, 208, 101, 36, 100, 43, 57, 202,
    242, 20, 99, 56, 28, 157, 124, 35, 113, 29, 160, 27, 211, 204, 210, 124, 113,
    46, 106, 43, 204, 96, 62, 77, 227, 137, 116, 73, 41, 72, 36, 203, 142, 235,
    64, 240, 48, 49, 64, 42, 41, 74, 214, 212, 207, 183, 42, 184, 210, 169, 211,
    140, 170, 176, 104, 105, 69, 193, 122, 168, 50, 123, 187, 89, 96, 90, 66, 65,
    107, 179, 89, 180, 119, 101, 120, 68, 63, 104, 234, 93, 227, 16, 15, 85, 209,
    129, 49, 15, 14, 86, 107, 55, 9, 120, 100, 121, 153, 145, 22, 178, 88, 179,
    197, 6, 196, 89, 88, 96, 135, 138, 136, 138, 215, 172, 218, 115, 219, 41, 42,
    81, 5, 195, 51, 57, 43, 61, 208, 171, 199, 41, 81, 38, 224, 53, 225, 24, 144,
    110, 105, 52, 66, 118, 229, 117, 227, 34, 234, 66, 107, 69, 10, 109, 151, 219,
    48, 235, 183, 62, 191, 142, 129, 126, 116, 111, 143, 7, 163, 246, 118, 117,
    50, 223, 222, 52, 94, 19, 141, 222, 221, 65, 196, 3, 197, 45, 220, 44, 156,
    70, 139, 188, 122, 245, 139, 71, 162, 145, 153, 159, 149, 170, 150, 122, 188,
    196, 206, 216, 92, 163, 144, 161, 164, 2, 167, 242, 141, 241, 0, 164, 37, 11,
    72, 12, 144, 145, 160, 12, 38, 13, 70, 63, 71, 31, 226, 111, 157, 158, 154,
    36, 101, 205, 203, 206, 165, 126, 209, 217, 98, 165, 97, 237, 220, 218, 237,
    239, 241, 210, 214, 169, 140, 171, 32, 241, 125, 237, 179, 86, 178, 180, 85,
    179, 181, 84, 180, 182, 83, 181, 194, 201, 182, 177, 137, 132, 184, 76, 183,
    185, 61, 184, 186, 57, 185, 216, 212, 186, 192, 214, 187, 139, 34, 156, 218,
    79, 237, 147, 123, 177, 45, 44, 4, 208, 201, 32, 98, 64, 129, 192, 213, 138,
    235, 59, 219, 141, 242, 97, 97, 2, 141, 240, 75, 235, 229, 24, 228, 31, 25,
    226, 230, 23, 229, 231, 22, 230, 232, 26, 231, 233, 112, 232, 244, 189, 243,
    189, 221, 190, 222, 28, 221, 223, 27, 222, 224, 29, 223, 225, 30, 224, 113,
    247, 225, 99, 60, 240, 213, 147, 215, 60, 20, 166, 192, 187, 213, 243, 112,
    244, 244, 233, 245, 245, 128, 188, 188, 114, 174, 134, 131, 220, 174, 217,
    236, 236, 198, 134, 215, 177, 58, 156, 143, 124, 25, 110, 7, 31, 228, 25, 264,
    356, 368, 0, 11, 267, 451, 452, 349, 267, 302, 269, 350, 357, 277, 350, 452,
    357, 299, 333, 297, 396, 175, 377, 381, 384, 382, 280, 347, 330, 269, 303,
    270, 151, 9, 337, 344, 278, 360, 424, 418, 431, 270, 304, 409, 272, 310, 407,
    322, 270, 410, 449, 450, 347, 432, 422, 434, 18, 313, 17, 291, 306, 375, 259,
    387, 260, 424, 335, 418, 434, 364, 416, 391, 423, 327, 301, 251, 298, 275,
    281, 4, 254, 373, 253, 375, 307, 321, 280, 425, 411, 200, 421, 18, 335, 321,
    406, 321, 320, 405, 314, 315, 17, 423, 426, 266, 396, 377, 369, 270, 322, 269,
    413, 417, 464, 385, 386, 258, 248, 456, 419, 298, 284, 333, 168, 417, 8, 448,
    346, 261, 417, 413, 285, 326, 327, 328, 277, 355, 329, 309, 392, 438, 381,
    382, 256, 279, 429, 360, 365, 364, 379, 355, 277, 437, 282, 443, 283, 281,
    275, 363, 395, 431, 369, 299, 297, 337, 335, 273, 321, 348, 450, 349, 359,
    446, 467, 283, 293, 282, 250, 458, 462, 300, 276, 383, 292, 308, 325, 283,
    276, 293, 264, 372, 447, 346, 352, 340, 354, 274, 19, 363, 456, 281, 426, 436,
    425, 380, 381, 252, 267, 269, 393, 421, 200, 428, 371, 266, 329, 432, 287,
    422, 290, 250, 328, 385, 258, 384, 446, 265, 342, 386, 387, 257, 422, 424,
    430, 445, 342, 276, 422, 273, 424, 306, 292, 307, 352, 366, 345, 268, 271,
    302, 358, 423, 371, 327, 294, 460, 331, 279, 294, 303, 271, 304, 436, 432,
    427, 304, 272, 408, 395, 394, 431, 378, 395, 400, 296, 334, 299, 6, 351, 168,
    376, 352, 411, 307, 325, 320, 285, 295, 336, 320, 319, 404, 329, 330, 349,
    334, 293, 333, 366, 323, 447, 316, 15, 315, 331, 358, 279, 317, 14, 316, 8,
    285, 9, 277, 329, 350, 253, 374, 252, 319, 318, 403, 351, 6, 419, 324, 318,
    325, 397, 367, 365, 288, 435, 397, 278, 344, 439, 310, 272, 311, 248, 195,
    281, 375, 273, 291, 175, 396, 199, 312, 311, 268, 276, 283, 445, 390, 373,
    339, 295, 282, 296, 448, 449, 346, 356, 264, 454, 337, 336, 299, 337, 338,
    151, 294, 278, 455, 308, 292, 415, 429, 358, 355, 265, 340, 372, 388, 390,
    466, 352, 346, 280, 295, 442, 282, 354, 19, 370, 285, 441, 295, 195, 248, 197,
    457, 440, 274, 301, 300, 368, 417, 351, 465, 251, 301, 389, 385, 380, 386,
    394, 395, 379, 399, 412, 419, 410, 436, 322, 387, 373, 388, 326, 2, 393, 354,
    370, 461, 393, 164, 267, 268, 302, 12, 386, 374, 387, 312, 268, 13, 298, 293,
    301, 265, 446, 340, 380, 385, 381, 280, 330, 425, 322, 426, 391, 420, 429,
    437, 393, 391, 326, 344, 440, 438, 458, 459, 461, 364, 434, 394, 428, 396,
    262, 274, 354, 457, 317, 316, 402, 316, 315, 403, 315, 314, 404, 314, 313,
    405, 313, 421, 406, 323, 366, 361, 292, 306, 407, 306, 291, 408, 291, 287,
    409, 287, 432, 410, 427, 434, 411, 372, 264, 383, 459, 309, 457, 366, 352,
    401, 1, 274, 4, 418, 421, 262, 331, 294, 358, 435, 433, 367, 392, 289, 439,
    328, 462, 326, 94, 2, 370, 289, 305, 455, 339, 254, 448, 359, 255, 446, 254,
    253, 449, 253, 252, 450, 252, 256, 451, 256, 341, 452, 414, 413, 463, 286,
    441, 414, 286, 258, 441, 258, 257, 442, 257, 259, 443, 259, 260, 444, 260,
    467, 445, 309, 459, 250, 305, 289, 290, 305, 290, 460, 401, 376, 435, 309,
    250, 392, 376, 411, 433, 453, 341, 464, 357, 453, 465, 343, 357, 412, 437,
    343, 399, 344, 360, 440, 420, 437, 456, 360, 420, 363, 361, 401, 288, 265,
    372, 353, 390, 339, 249, 339, 448, 255,
];
const right_eye = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 33];
const left_eye = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 362];
const mouth = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78];
const nose1 = [240, 97, 2, 326, 327];
const nose2 = [2, 94, 19, 1, 4, 5, 195, 197, 6, 168, 8, 107, 66, 105, 63, 70];
const nose3 = [8, 336, 296, 334, 293, 300];
const ghost_mask_array = [right_eye, left_eye, mouth, nose1, nose2, nose3];

function set_slick(orientation, slides, arrows) {
    let prev_string = '<button type="button" class="ref_btn"><img src="/images/Thumbs/arrow_' + arrows[0] + '.png" alt="PREV"></button>';
    let next_string = '<button type="button" class="ref_btn"><img src="/images/Thumbs/arrow_' + arrows[1] + '.png" alt="NEXT"></button>';

    leftSlick.slick('slickSetOption', 'slidesToShow', slides);
    rightSlick.slick('slickSetOption', 'slidesToShow', slides);
    leftSlick.slick('slickSetOption', 'vertical', orientation);
    rightSlick.slick('slickSetOption', 'vertical', orientation);
    leftSlick.slick('slickSetOption', 'prevArrow', prev_string)
    leftSlick.slick('slickSetOption', 'nextArrow', next_string)
    rightSlick.slick('slickSetOption', 'prevArrow', prev_string)
    rightSlick.slick('slickSetOption', 'nextArrow', next_string)
    leftSlick.slick('refresh');
    rightSlick.slick('refresh');
}

function init_slicks() {
    if (leftSlick.length) {
        leftSlick.slick({
            dots: false,
            infinite: false,
            speed: 300,
            focusOnSelect: false,
            slidesToShow: 3.5,
            slidesToScroll: 3,
            //arrows: false,
            vertical: true,
            draggable: true,
            asNavFor: '.asnav1Class',
            prevArrow: "<button type='button' class='ref_btn'><img src='/images/Thumbs/arrow_up.png' alt='UP'></button>",
            nextArrow: "<button type='button' class='ref_btn'><img src='/images/Thumbs/arrow_down.png' alt='DOWN'></button>"

        });
        rightSlick.slick({
            dots: false,
            infinite: false,
            speed: 300,
            focusOnSelect: false,
            slidesToShow: 3.5,
            slidesToScroll: 3,
            vertical: true,
            //        arrows: true,
            draggable: true,
            asNavFor: '.asnav2Class',
            prevArrow: "<button type='button' class='ref_btn'><img src='/images/Thumbs/arrow_up.png' alt='UP'></button>",
            nextArrow: "<button type='button' class='ref_btn'><img src='/images/Thumbs/arrow_down.png' alt='DOWN'></button>"
        });

        // On init
        $(".select_ch").each(function (index, el) {
            leftSlick.slick('slickAdd', "<div>" + el.innerHTML + "</div>");
        });
        $(".show_morph").each(function (index, el) {
            rightSlick.slick('slickAdd', "<div>" + el.innerHTML + "</div>");
        });
    }
}

function window_size() {
    let orientation, arrows, areas, columns, rows;
    const width = container.offsetWidth
    let slides = (width <= 700 && width >= 600) ? 6 : (width < 600 && width >= 450) ? 5 : (width < 450) ? 4 : 3.5
    if (width <= 700) {
        areas = '"main_view main_view main_view main_view" "ref_btns ref_btns ref_btns ref_btns" "morph_btns morph_btns morph_btns morph_btns"';
        columns = '25% 25% 25% 25%';
        rows = '70% 15% 15%';
        container_center.style.width = container_center.style.maxHeight = Math.round(container.offsetHeight * 0.7 - hints.offsetHeight - 40) + 'px';
        // container_center.style.maxHeight;
        container_right.style.maxHeight = container_left.style.maxHeight = Math.round(container.offsetHeight * 0.2) + 'px';
        container_left.style.display = 'inline-flex'
        container_right.style.display = 'inline-flex'
        arrows = ['left', 'right']
        orientation = false

    } else {
        areas = '"ref_btns main_view main_view morph_btns" "ref_btns main_view main_view morph_btns" "ref_btns main_view main_view morph_btns"';
        columns = '20% 30% 30% 20%';
        rows = '40% 40% 20%';
        container_center.style.maxHeight = Math.round(container.offsetHeight - hints.offsetHeight - 40) + 'px';
        container_center.style.width = Math.round(container.offsetWidth * 0.6 - 20) + 'px';
        container_left.style.display = container_right.style.display = 'block';
        arrows = ['up', 'down'];
        orientation = true
    }
    canvas.style.maxHeight = hints.style.maxWidth = container_center.style.maxHeight
    container.style.gridTemplateAreas = areas
    container.style.gridTemplateColumns = columns;
    container.style.gridTemplateRows = rows;

    container_center.style.maxWidth = Math.round(container.offsetWidth - 20) + 'px';
    set_slick(orientation, slides, arrows)
}

function path_adjusted(url) {
    if (!/^\w+:/i.test(url)) {
        url = url.replace(/^(\.?\/?)([\w@])/, "$1js/$2")
    }
    return url
}

function extract_index(path) {
    const fileName = path.split('/').pop()
    const replaced = fileName.replace(/\D/g, ''); // 👉️ '123'
    let num;
    if (replaced !== '') {
        num = Number(replaced) - 1; // 👉️ 123
    }
    return num
}

function setMorphsButtons(img) {
    for (let i = 0; i < m_all_btns.length; i++) {
        m_all_btns[i].firstChild.src = img;
    }
}

function drawOnCanvas(my_img) {
    ref_img.src = my_img;
    ref_img.onload = function () {
        let dsize = new cv.Size(container_center.offsetWidth, container_center.offsetWidth);
        let img = cv.imread(ref_img)
        cv.resize(img, img, dsize, 0, 0, cv.INTER_AREA);
        cv.imshow(canvas, img);
    }
}

function setButtonClick(button, action) {
    button.onclick = action;
}

/**
 * Solution options.
 */
const solutionOptions = {
    selfieMode: true,
    enableFaceGeometry: false,
    maxNumFaces: 1,
    refineLandmarks: false,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
};


function onResults() {
    const results = mpFaceMesh.Results
        /
        // Draw the overlays.
        context.save();
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(
        results.image, 0, 0, canvas.width, canvas.height);
    if (results.multiFaceLandmarks) {
        for (const landmarks of results.multiFaceLandmarks) {
            drawingUtils.drawConnectors(
                context, landmarks, mpFaceMesh.FACEMESH_TESSELATION,
                {color: '#C0C0C070', lineWidth: 1});
            drawingUtils.drawConnectors(
                context, landmarks, mpFaceMesh.FACEMESH_RIGHT_EYE,
                {color: '#FF3030'});
            drawingUtils.drawConnectors(
                context, landmarks, mpFaceMesh.FACEMESH_RIGHT_EYEBROW,
                {color: '#FF3030'});
            drawingUtils.drawConnectors(
                context, landmarks, mpFaceMesh.FACEMESH_LEFT_EYE,
                {color: '#30FF30'});
            drawingUtils.drawConnectors(
                context, landmarks, mpFaceMesh.FACEMESH_LEFT_EYEBROW,
                {color: '#30FF30'});
            drawingUtils.drawConnectors(
                context, landmarks, mpFaceMesh.FACEMESH_FACE_OVAL,
                {color: '#E0E0E0'});
            drawingUtils.drawConnectors(
                context, landmarks, mpFaceMesh.FACEMESH_LIPS, {color: '#E0E0E0'});
            if (solutionOptions.refineLandmarks) {
                drawingUtils.drawConnectors(
                    context, landmarks, mpFaceMesh.FACEMESH_RIGHT_IRIS,
                    {color: '#FF3030'});
                drawingUtils.drawConnectors(
                    context, landmarks, mpFaceMesh.FACEMESH_LEFT_IRIS,
                    {color: '#30FF30'});
            }
        }
    }
    context.restore();
}

async function init() {
    let all_btns_indices = [];
    /* INITIALIZE FACE LANDMARK DETECTOR */
    detector = await faceLandmarksDetection.createDetector(model, detectorConfig);

    /* INITIALIZE UI */
    init_slicks()
    window_size()
    drawOnCanvas(start_view)
    console.log('UI INITIALIZED')
    /* INITIALIZE SEND EMAIL POPUP */
    const validRegex = /^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*$/;
    popupButton.firstChild.src = send_logo
    resetButton.firstChild.src = reset_logo
    resetButton.addEventListener('click', () => {
        fetch("/delete_morphs", {
            method: 'POST',
            headers: {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
            body: JSON.stringify({'morphs': 'delete'}),
        })
            .then((res) => {
                if (!res.ok) {
                    throw new Error(`HTTP error: ${res.status}`);
                }
                return res.json();
            })
            .then((json) => {
                drawOnCanvas(start_view);
                setMorphsButtons(default_morph);
            })
            .catch((err) => console.error(`Fetch problem: ${err.message}`));
        popupWindow.style.display = 'none';
    })
    popupButton.addEventListener('click', () => {
        popupWindow.style.display = 'block';
    });
    sendEmailButton.addEventListener('click', () => {
        // Get the email address from the input field
        const mailToAddress = emailInput.value;
        // const files = fs.promises.readdir(morphs_path);
        if (mailToAddress.match(validRegex)) {
            fetch("/morphs_to_send", {
                method: 'POST',
                headers: {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
                body: JSON.stringify({'mail': mailToAddress}),
            })
                .then((res) => {
                    if (!res.ok) {
                        throw new Error(`HTTP error: ${res.status}`);
                    }
                    return res.json();
                })
                .then((json) => {
                    drawOnCanvas(start_view);
                    setMorphsButtons(default_morph);
                })
                .catch((err) => console.error(`Fetch problem: ${err.message}`));

            // Close the popup window
            popupWindow.style.display = 'none';
        } else {
            if (confirm("Your input is not a valid email address.\n" +
                "Press Cancel to retry or Ok to reset the game!")) {

                fetch("/delete_morphs", {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
                    body: JSON.stringify({'morphs': 'delete'}),
                })
                    .then((res) => {
                        if (!res.ok) {
                            throw new Error(`HTTP error: ${res.status}`);
                        }
                        return res.json();
                    })
                    .then((json) => {
                        drawOnCanvas(start_view);
                        setMorphsButtons(default_morph);
                    })
                    .catch((err) => console.error(`Fetch problem: ${err.message}`));
                popupWindow.style.display = 'none';
            } else {
                console.log('Retry')
            }
            return false;

        }
    });
    console.log('EMAIL INITIALIZED')
    /* INITIALIZE PAINTINGS' FACE OBJECTS */
    let start_message = {'start': 'paintings'}
    await fetch("/init", {
        method: 'POST',
        headers: {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
        body: JSON.stringify(start_message)
    })
        .then((res) => {
            if (!res.ok) {
                throw new Error(`HTTP error: ${res.status}`);
            }
            return res.json();

        })
        .then((json) => {
            face_arr = json.body
        })
    console.log('PAINTINGS OBJ INITIALIZED')
    /* INITIALIZE SLICKs BUTTONS AND INTERACTION */
    let all_btns = container_left.querySelectorAll("div.slick-slide >button");
    m_all_btns = container_right.querySelectorAll("div.slick-slide >button");

    all_btns.forEach(function (item, index, arr) {
        all_btns_indices.push(arr[index].parentNode.getAttribute('data-slick-index'))
    })

    for (let j = 0; j < all_btns.length; j++) {
        setButtonClick(all_btns[j], function () {
            let _this = this;
            selected = extract_index(_this.firstChild.src)
            var url = protocol + '//' + host;
            if (port) {
                url += ':' + port;
            }
            url += '/' + face_arr[selected]['src'];
            ref_img.src = url;
            // drawOnCanvas(ref_img.src)
            ref_img.onload = function () {
                if (selected !== -1) {
                    morphed = ''
                    detect_interval = setInterval(match_faces, 1000 / 30);
                    console.log('start interval')
                }
            }
        });

        setButtonClick(m_all_btns[j], function () {
            selected = -1
            let _this = this;
            let slide_id = all_btns_indices[j]
            const m_selected = extract_index(_this.firstChild.src)
            if (detect_interval) {
                clearInterval(detect_interval)
                reset_bar()
            }
            ref_img.src = (isNaN(m_selected) === false && slide_id !== extract_index(ref_img.src)) ? _this.firstChild.src : default_view;
            drawOnCanvas(ref_img.src)
        })
    }
    console.log('INTERACTION INITIALIZED')
}

function reset_bar() {
    [percent_x, percent_y, percent_z].forEach(bar => {
        bar.style.width = bar.innerHTML = '';
    });
}

function update_bar() {
    if (morphed === '') {
        let matchings = [
            100 - Math.abs(face_arr[selected].angles[0] - cam_face.angles[0]),
            100 - Math.abs(face_arr[selected].angles[1] - cam_face.angles[1]),
            100 - Math.abs(face_arr[selected].angles[2] - cam_face.angles[2])
        ];

        [percent_x, percent_y, percent_z].forEach((element, index) => {
            element.style.width = matchings[index] + '%';
            element.innerHTML = matchings[index] + '%';
        });
    }
}

/* FACE DATA CALCULATIONS */
function Face(which, image) {
    this.which = which;
    this.image = image;
    this.type = typeof (image);
    this.src = image.src;
    this.w = image.naturalWidth;
    this.h = image.naturalHeight;
    this.points = [];
    this.bb = [];
    this.hull = [];
    this.n_points = []
    this.expression = []
    this.normalize_array = function (pix_points) {
        let n_array = [...pix_points];
        for (let p in n_array) {
            n_array[p] = [Math.round(n_array[p][0]) / this.w, Math.round(n_array[p][1]) / this.h]
        }
        return n_array
    }
}

async function calc_lmrks(image, which) {
    let landmarks = []
    let points_2d = []
    let bb = {}
    let angles
    let faces = await detector.estimateFaces(image, {flipHorizontal: false});
    if (faces.length >> 0) {
        const keypoints = faces[0].keypoints;
        for (let land = 0; land < keypoints.length; land++) {
            let x = Math.round(keypoints[land].x);
            let y = Math.round(keypoints[land].y);
            let z = Math.round(keypoints[land].z);
            landmarks.push([x, y, z])
            points_2d.push([x, y])
        }
        bb.xMin = Math.round(faces[0].box.xMin)
        bb.xMax = Math.round(faces[0].box.xMax)
        bb.yMin = Math.round(faces[0].box.yMin)
        bb.yMax = Math.round(faces[0].box.yMax)
        bb.width = Math.round(faces[0].box.width)
        bb.height = Math.round(faces[0].box.height)
        bb.center = [bb.xMin + Math.round(bb.width / 2), bb.yMin + Math.round(bb.height / 2)]

        if (landmarks !== []) {
            angles = await getHeadAngles(landmarks, which)
        }
    } else {
        console.log('Cannot calculate landmarks of image ' + which)
    }
    return [landmarks, points_2d, bb, angles]
}


function calcHull(points) {
    if (points.length <= 1)
        return points.slice();
    let upperHull = [];
    for (let i = 0; i < points.length; i++) {
        let p = points[i];
        while (upperHull.length >= 2) {
            let q = upperHull[upperHull.length - 1];
            let r = upperHull[upperHull.length - 2];
            if ((q[0] - r[0]) * (p[1] - r[1]) >= (q[1] - r[1]) * (p[0] - r[0]))
                upperHull.pop();
            else
                break;
        }
        upperHull.push(p);
    }
    upperHull.pop();
    let lowerHull = [];
    for (let i = points.length - 1; i >= 0; i--) {
        let p = points[i];
        while (lowerHull.length >= 2) {
            let q = lowerHull[lowerHull.length - 1];
            let r = lowerHull[lowerHull.length - 2];
            if ((q[0] - r[0]) * (p[1] - r[1]) >= (q[1] - r[1]) * (p[0] - r[0]))
                lowerHull.pop();
            else
                break;
        }
        lowerHull.push(p);
    }
    lowerHull.pop();
    if (upperHull.length === 1 && lowerHull.length === 1 && upperHull[0].x === lowerHull[0].x && upperHull[0].y === lowerHull[0].y) {
        return upperHull;
    } else {
        return upperHull.concat(lowerHull);
    }
}

function convex_hull(points, boundingbox) {
    let p1, p2, p3, p4
    for (let pt in points) {
        if (points[pt][0] === boundingbox.xMin) {
            p1 = points[pt]
        } else if (points[pt][0] === boundingbox.xMax) {
            p2 = points[pt]
        } else if (points[pt][1] === boundingbox.yMin) {
            p3 = points[pt]
        } else if (points[pt][1] === boundingbox.yMax) {
            p4 = points[pt]
        }
    }
    const poly = [p1, p2, p3, p4];

    // Use the filter method to remove the points that are inside the polygon
    let new_arr = points.filter((pt) => !pointInPolygon(poly, pt));

    // Add the bounding box points to the new array
    new_arr = [...new_arr, ...poly];

    // Sort the points in the new array
    new_arr.sort(point_comparator);

    // Calculate the convex hull of the new array
    let hull = calcHull(new_arr);

    // Use the map method to transform the hull array
    return hull.map((h) => points.indexOf(h));
}

async function getHeadAngles(keypoints, which) {
    // V: 10, 152; H: 226, 446
    this.faceVerticalCentralPoint = [
        0,
        (keypoints[10][1] + keypoints[152][1]) * 0.5,
        (keypoints[10][2] + keypoints[152][2]) * 0.5,
    ];
    const verticalAdjacent =
        keypoints[10][2] - this.faceVerticalCentralPoint[2];
    const verticalOpposite =
        keypoints[10][1] - this.faceVerticalCentralPoint[1];
    const verticalHypotenuse = this.l2Norm([
        verticalAdjacent,
        verticalOpposite,
    ]);
    const verticalCos = verticalAdjacent / verticalHypotenuse;

    this.faceHorizontalCentralPoint = [
        (keypoints[226][0] + keypoints[446][0]) * 0.5,
        0,
        (keypoints[226][2] + keypoints[446][2]) * 0.5,
    ];
    const horizontalAdjacent =
        keypoints[226][2] - this.faceHorizontalCentralPoint[2];
    const horizontalOpposite =
        keypoints[226][0] - this.faceHorizontalCentralPoint[0];
    const horizontalHypotenuse = this.l2Norm([
        horizontalAdjacent,
        horizontalOpposite,
    ]);
    const horizontalCos = horizontalAdjacent / horizontalHypotenuse;
    let first = Math.acos(verticalCos) * (180 / Math.PI);
    let second = Math.acos(horizontalCos) * (180 / Math.PI);
    if (which === 'cam') {
        first = Math.round(normalize(first, {
            'actual': {'lower': 55, 'upper': 115},
            'desired': {'lower': 82, 'upper': 95}
        }))
        second = Math.round(normalize(second, {
            'actual': {'lower': 50, 'upper': 120},
            'desired': {'lower': 120, 'upper': 69}
        }))
    }
    let pleft = keypoints[133]
    let pright = keypoints[362]
    let third = Math.atan2(pright[1] - pleft[1], pright[0] - pleft[0]) * (180 / Math.PI)
    return [first, second, Math.round(third)];
}

function pointInPolygon(polygon, point) {
    //A point is in a polygon if a line from the point to infinity crosses the polygon an odd number of times
    let odd = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; i++) {
        if (((polygon[i][1] > point[1]) !== (polygon[j][1] > point[1])) &&
            (point[0] < ((polygon[j][0] - polygon[i][0]) * (point[1] - polygon[i][1]) / (polygon[j][1] - polygon[i][1]) + polygon[i][0]))) {
            odd = !odd;
        }
        j = i;
    }
    return odd;
}

function l2Norm(vec) {
    let norm = 0;
    for (const v of vec) {
        norm += v * v;
    }
    return Math.sqrt(norm);
}

function normalize(value, bounds) {
    return bounds['desired']['lower'] + (value - bounds['actual']['lower']) *
        (bounds['desired']['upper'] - bounds['desired']['lower']) /
        (bounds['actual']['upper'] - bounds['actual']['lower'])
}

function point_comparator(a, b) {
    if (a[0] < b[0])
        return -1;
    else if (a[0] > b[0])
        return +1;
    else if (a[1] < b[1])
        return -1;
    else if (a[1] > b[1])
        return +1;
    else
        return 0;
}

/* DRAW GHOST MASK ON PAINTING FACE*/
function draw_lines(img, arr) {
    const line_color = new cv.Scalar(255, 255, 255, 120);
    for (let i = 0; i < arr.length - 1; i++) {
        const point1 = new cv.Point(cam_face.points[arr[i]][0], cam_face.points[arr[i]][1]);
        const point2 = new cv.Point(cam_face.points[arr[i + 1]][0], cam_face.points[arr[i + 1]][1]);
        cv.line(img, point1, point2, line_color, 1);
    }
    if (arr.length % 2 === 1) {
        const lastPoint = new cv.Point(cam_face.points[arr[arr.length - 1]][0], cam_face.points[arr[arr.length - 1]][1]);
        const secondLastPoint = new cv.Point(cam_face.points[arr[arr.length - 2]][0], cam_face.points[arr[arr.length - 2]][1]);
        cv.line(img, lastPoint, secondLastPoint, line_color, 1);
    }
}

function draw_mask_on_ref() {
    // Get bounding boxes for face in reference image and camera image
    const ref_bb = face_arr[selected].bb
    const bb_cam_rect = new cv.Rect(cam_face.bb.xMin, cam_face.bb.yMin, cam_face.bb.width, cam_face.bb.height)
    const bb_ref_rect = new cv.Rect(ref_bb.xMin, ref_bb.yMin, ref_bb.width, ref_bb.height);

    // Capture frame from video stream and create copy for drawing ghost lines
    let cap = new cv.VideoCapture(video);
    let cam_source = new cv.Mat(video.height, video.width, cv.CV_8UC4)
    cap.read(cam_source)
    let ghost_source = new cv.Mat(video.height, video.width, cv.CV_8UC4)

    // draw mask on ghost_source
    for (const arr of ghost_mask_array) {
        draw_lines(ghost_source, arr);
    }

    // Extract region of interest (ROI) from camera frame using camera bounding box
    let cam_roi = ghost_source.roi(bb_cam_rect)
    cam_source.delete();
    ghost_source.delete();

    // Load reference image and extract ROI using reference bounding box
    const ref = cv.imread(ref_img)
    const ref_roi = ref.roi(bb_ref_rect)

    // Get sizes for resizing images later
    const cameraROISize = new cv.Size(cam_roi.cols, cam_roi.rows)
    const referenceROISize = new cv.Size(ref_roi.cols, ref_roi.rows)
    const last_size = new cv.Size(canvas.width, canvas.height)

    // Blur ghost mask
    cv.GaussianBlur(cam_roi, cam_roi, new cv.Size(11, 11), 20, 20, cv.BORDER_DEFAULT)

    // Convert camera ROI to grayscale
    let cam_roi_gray = new cv.Mat()
    cv.cvtColor(cam_roi, cam_roi_gray, cv.COLOR_RGBA2GRAY, 0)
    cam_roi.delete();

    // Generate convex hull mask
    let convexHullMat = hull_mask()
    let cam_mask = convexHullMat.roi(bb_cam_rect);
    convexHullMat.delete();

    // Convert mask to grayscale and combine with camera ROI
    cv.cvtColor(cam_mask, cam_mask, cv.COLOR_RGBA2GRAY, 0)
    let ghost_mask = new cv.Mat()
    cv.bitwise_and(cam_roi_gray, cam_roi_gray, ghost_mask, cam_mask)
    cam_mask.delete();
    cam_roi_gray.delete();

    // Flip mask horizontally and convert to RGB and RGBA formats
    cv.flip(ghost_mask, ghost_mask, 1)
    cv.cvtColor(ghost_mask, ghost_mask, cv.COLOR_GRAY2RGB, 0)
    cv.cvtColor(ghost_mask, ghost_mask, cv.COLOR_RGB2RGBA, 0)

    // Apply ghost mask over reference image
    cv.resize(ref_roi, ref_roi, cameraROISize, 0, 0, cv.INTER_LINEAR);
    let sum = new cv.Mat();
    cv.add(ref_roi, ghost_mask, sum);
    cv.add(sum, ghost_mask, sum)
    ghost_mask.delete();
    ref_roi.delete();
    cv.resize(sum, sum, referenceROISize, 0, 0, cv.INTER_LINEAR)

    // Copy result to new image and apply to canvas
    let result = ref.clone();
    ref.delete();
    sum.copyTo(result.roi(bb_ref_rect));
    sum.delete();
    cv.resize(result, result, last_size, 0, 0, cv.INTER_LINEAR)
    cv.imshow(canvas, result);
    result.delete()
}

function hull_mask() {
    const hull_points = cam_face.hull.map((id) => [cam_face.n_points[id][0] * cam_face.w,
        cam_face.n_points[id][1] * cam_face.h,]);
    let hullMat = cv.matFromArray(hull_points.length, 1, cv.CV_32SC2, hull_points.flat());
    let hulls = new cv.MatVector();
    hulls.push_back(hullMat);
    let convexHullMat = cv.Mat.zeros(cam_face.w, cam_face.h, cv.CV_8UC3);
    cv.drawContours(convexHullMat, hulls, 0, [255, 255, 255, 0], -1, 8);
    hullMat.delete();
    hulls.delete()
    return convexHullMat
}

/* MATCHING FUNCTIONS */
async function match_faces() {
    if (selected >= 0) {
        cam_face = new Face('cam', video);
        cam_face.w = video.width;
        cam_face.h = video.height;

        try {
            const [lmrks, points, bb, angles] = await calc_lmrks(video, 'cam');
            cam_face.lmrks = lmrks;
            cam_face.points = points;
            cam_face.bb = bb;
            cam_face.angles = angles;
            cam_face.hull = convex_hull(points, bb);
            cam_face.n_points = cam_face.normalize_array(points);
            cam_face.expression = check_expression(points);
            update_bar();

            draw_mask_on_ref();
            check_and_swap(cam_face.angles, face_arr[selected].angles);
        } catch (error) {
            console.error(error);
            // handle the error here, e.g. display an error message to the user
        }
    }
}

async function check_and_swap(angles_cam, angles_ref) {
    const delta = 8;
    const [angle1_cam, angle2_cam, angle3_cam] = angles_cam;
    const [angle1_ref, angle2_ref, angle3_ref] = angles_ref;
    if (
        angle1_cam >= angle1_ref - delta &&
        angle1_cam <= angle1_ref + delta &&
        angle2_cam >= angle2_ref - delta &&
        angle2_cam <= angle2_ref + delta &&
        angle3_cam >= angle3_ref - delta / 2 &&
        angle3_cam <= angle3_ref + delta / 2
    ) {
        console.log('match1')
        if (cam_face.expression.toString() === face_arr[selected].expression.toString()) {
            console.log('match2')
            clearInterval(detect_interval)
            morphed = ''
            // Draw image on canvas
            ctx.drawImage(video, 0, 0, cam_face.w, cam_face.h);
            // Create data URL
            const data_url = cam_canvas.toDataURL('image/jpeg', 0.5);
            // Clear canvas
            ctx.clearRect(0, 0, cam_face.w, cam_face.h);
            // Set cam_face image property
            cam_face.image = data_url

            try {
                const objs = {selected, c_face: cam_face.image};
                const data_json = JSON.stringify(objs);

                // Send data to server
                const res = await fetch('/info', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
                    body: data_json,
                });

                if (!res.ok) {
                    throw new Error(`HTTP error: ${res.status}`);
                }

                const json = await res.json();

                // Extract information from server response
                const rel_path = json.relative_path;
                const abs_path = json.absolute_path;
                const user_id = json.user_id;
                const folder = 'public';
                const folder_id = abs_path.indexOf(folder);
                const sub_path = abs_path.slice(folder_id + folder.length);

                // Construct morphed URL
                const morphed_base = `${protocol}//${host}`;
                const morphed_port = port ? `:${port}` : '';
                morphed = `${morphed_base}${morphed_port}${sub_path}`;

                // Update button image and canvas
                const id = extract_index(rel_path);
                m_all_btns[id].firstChild.src = morphed;
                drawOnCanvas(morphed);
                reset_bar();
                selected = -1;
            } catch (err) {
                console.error(`Fetch problem: ${err.message}`);
            }
        }
    }
}

function check_expression(landmarks) {
    let l_e, r_e, lips

    function calc_division(p1, p2, p3, p4) {
        const p4_p3 = ((p4[0] - p3[0]) ** 2 + (p4[1] - p3[1]) ** 2) ** 0.5
        const p2_p1 = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
        return p4_p3 / p2_p1
    }

    // l_eye
    const l_division = calc_division(landmarks[362], landmarks[263], landmarks[386], landmarks[374])
    l_division <= 0.1 ? l_e = 'closed' : l_e = 'opened';
    // r_eye
    const r_division = calc_division(landmarks[33], landmarks[133], landmarks[159], landmarks[145])
    r_division <= 0.1 ? r_e = 'closed' : r_e = 'opened';
    // Mouth
    const lips_division = calc_division(landmarks[78], landmarks[308], landmarks[13], landmarks[14])
    lips_division < 0.1 ? lips = 'closed' : lips_division > 0.5 ? lips = 'full opened' : lips = 'opened';
    return [l_e, r_e, lips]
}

function resize_all() {
    window_size()
    ref_img.src = ref_img.src
    if (selected >= 0) {
        draw_mask_on_ref()
    }
}

const accessCamera = () => {
    navigator.mediaDevices
        .getUserMedia({
            video: {facingMode: 'user', width: 1280, height: 960},
            audio: false,
        })
        .then((stream) => {
            video.srcObject = stream;
        })
        .catch(function (error) {
            console.log("Something went wrong!", error);
        });
};

function opencvIsReady() {
    console.log('OPENCV.JS READY')
    init().then(r => console.log('ALL IS INITIALIZED'));
    accessCamera();
}

window.addEventListener('resize', resize_all, false);
video.addEventListener('loadeddata', function () {
    body.classList.add('loaded')
})
window.addEventListener('beforeunload', function (event) {
    const confirmationMessage = 'Are you sure you want to delete the folder?';
    event.preventDefault();
    event.returnValue = confirmationMessage;
    setTimeout(function () {
        if (!event.returnValue) {
            // The user clicked the "Cancel" button
            console.log('User canceled the action');
        } else {
            // The user clicked the "OK" button
            console.log('User confirmed the action');
            // Send a request to the server to delete the folder
            fetch('/folder', {
                method: 'DELETE'
            }).then(response => {
                if (response.ok) {
                    console.log('Folder deleted successfully');
                } else {
                    console.error('Error deleting folder');
                }
            }).catch(error => {
                console.error(error);
            });
        }
    }, 0);
});


