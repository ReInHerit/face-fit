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
const context = canvas.getContext('2d');
/* POPUP */
const popupButton = document.getElementById('popup-button');
const resetButton = document.getElementById('reset-button');
const popupWindow = document.getElementById('popup-window');
const emailInput = document.getElementById('email-input');
const sendEmailButton = document.getElementById('send-email-button');
const camera_width = 1280;
const camera_height = 960;
const framerate = 5;
const interval = 1000 / framerate;
const port = window.location.port;
const host = window.location.hostname;
const protocol = window.location.protocol;
const url_base = `${protocol}//${host}`;
const url_port = port ? `:${port}` : '';
// let cam_face = new Face('cam', video);
let selected, morphed_btns;
let face_arr = []
let morphed = ''
const model = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;
const detectorConfig = {
    runtime: 'tfjs',//'mediapipetfjs',
    solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh',
    min_tracking_confidence: 0.5,
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: false,
}
// const morphs_path = '../morphs'
const default_view = '../images/Thumbs/default_view.jpg'
const default_morph = '../images/Thumbs/morph_thumb.jpg'
const send_logo = '../images/Thumbs/send.png'
const reset_logo = '../images/Thumbs/reset.png'
const start_view = '../images/Thumbs/start_view.jpg'

const leftSlick = $("#ref_btns");
const rightSlick = $("#morph_btns")

const right_eye = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 33];
const left_eye = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 362];
const mouth = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78];
const nose1 = [240, 97, 2, 326, 327];
const nose2 = [2, 94, 19, 1, 4, 5, 195, 197, 6, 168, 8, 107, 66, 105, 63, 70];
const nose3 = [8, 336, 296, 334, 293, 300];
const full_indices_set = new Set([...right_eye, ...left_eye, ...mouth, ...nose1, ...nose2, ...nose3, 10, 152, 226, 446]);
const ghost_mask_array = [right_eye, left_eye, mouth, nose1, nose2, nose3];
let  ref_roi, ref, ref_size, result, bb_ref_rect, ref_roi_size, matchInterval, detector;
const line_color = new cv.Scalar(255, 255, 255, 120);
let isMatching = false;
let cam_points = {}
let cam_bb = {}
let cam_angles = []
let cam_expression = []

let dsize = new cv.Size(container_center.offsetWidth, container_center.offsetWidth);

/* UI FUNCTIONS*/
function init_slicks(window_aspect_ratio) {
    const sliders = [leftSlick, rightSlick];
    const isVertical = window_aspect_ratio === 'vertical'
    let arrows = isVertical ? ['left', 'right'] : ['up', 'down'];
    const prevArrowString = `<button type="button" class="slick-prev"><img src="/images/Thumbs/arrow_${arrows[0]}.png" alt="PREV"></button>`;
    const nextArrowString = `<button type="button" class="slick-next"><img src="/images/Thumbs/arrow_${arrows[1]}.png" alt="NEXT"></button>`;
    // Calculate slide width and count once
    const [slides, value] = calculateSlideWidthAndCount(isVertical);

    // Define a common function for setting arrow properties
    function setArrowProperties(slider, value) {
        slider.find('.button-image img').css('width', (value - 5) + 'px');
        if (!isVertical) {
            slider.find('.button-image img').css('max-height', (value - 5) + 'px');
        }
    }

    sliders.forEach((slider, index) => {
        slider.on('init', function (event, slick) {
            // console.log('init_slicks', index, slick, value)
            setArrowProperties(slider, value);
        }).slick({
            infinite: false,
            autoplay: false,
            slidesToShow: slides,
            slidesToScroll: slides,
            dots: false,
            vertical: isVertical,
            draggable: true,
            asNavFor: `.asnav${index + 1}Class`,
            prevArrow: prevArrowString,
            nextArrow: nextArrowString
        });
        slider.on('beforeChange', function (event, slick, currentSlide, nextSlide) {
            const $prevArrow = $(slick.$prevArrow);
            const $nextArrow = $(slick.$nextArrow);

            // Disable the previous arrow when reaching the start of the slider
            $prevArrow.toggleClass('disabled', nextSlide === 0);

            // Disable the next arrow when reaching the end of the slider
            $nextArrow.toggleClass('disabled', nextSlide === slick.slideCount - 1);
        });
    });

    console.log('init_slicks',)
}

function updateSlider() {
    const width = container.offsetWidth;
    const height = container.offsetHeight;
    const aspect = width / height;
    const isVertical = (aspect <= 1);
    let orientation, areas, arrows, columns, rows, area;

    const [slides, value] = calculateSlideWidthAndCount(isVertical);
    // console.log('updateSlider', slides, value);

    if (isVertical) {
        orientation = 'vertical';
        arrows = ['left', 'right'];
        columns = '25vw 25vw 25vw 25vw';
        rows = '70% 15% 15%';
        area = '"main_view main_view main_view main_view" "ref_btns ref_btns ref_btns ref_btns" "morph_btns morph_btns morph_btns morph_btns"';

        container_center.style.width = container_center.style.maxHeight = `${Math.round(height * 0.7 - hints.offsetHeight - 40)}px`;
        container_right.style.maxHeight = container_left.style.maxHeight = `${Math.round(height * 0.2)}px`;
        container_right.style.flexDirection = container_left.style.flexDirection = 'row';
        canvas.style.maxHeight = hints.style.maxWidth = container_center.style.maxHeight;
    } else {
        orientation = 'horizontal';
        arrows = ['up', 'down'];
        columns = '20vw 30vw 30vw 20vw';
        rows = '40% 40% 20%';
        area = '"ref_btns main_view main_view morph_btns" "ref_btns main_view main_view morph_btns" "ref_btns main_view main_view morph_btns"';

        container_center.style.maxHeight = `${Math.round(height - hints.offsetHeight - 40)}px`;
        container_right.style.maxHeight = container_left.style.maxHeight = `${Math.round(height)}px`;
        container_center.style.width = `${Math.round(width * 0.6 - 20)}px`;
        canvas.style.maxHeight = hints.style.maxWidth = `${height - hints.offsetHeight}px`;
        container_right.style.flexDirection = container_left.style.flexDirection = 'column';
    }

    const prevString = `<button type="button" class="slick-prev"><img src="/images/Thumbs/arrow_${arrows[0]}.png" alt="PREV"></button>`;
    const nextString = `<button type="button" class="slick-next"><img src="/images/Thumbs/arrow_${arrows[1]}.png" alt="NEXT"></button>`;

    [leftSlick, rightSlick].forEach((slider) => {
        slider.slick('slickSetOption', {
            slidesToShow: slides,
            slidesToScroll: slides,
            vertical: orientation === 'horizontal',
            prevArrow: prevString,
            nextArrow: nextString
        });

        slider.find('.button-image img').css('width', `${value - 5}px`);
        if (!isVertical) {
            slider.find('.button-image img').css('max-height', `${value - 5}px`);
        }
        slider.slick('refresh');
    })

    container.style.gridTemplateAreas = area;
    container.style.gridTemplateColumns = columns;
    container.style.gridTemplateRows = rows;
    container_center.style.maxWidth = `${Math.round(width - 20)}px`;
}

function calculateSlideWidthAndCount(vertical) {
    // Ensure that 'container' is a valid DOM element
    if (!(container instanceof HTMLElement)) {
        throw new Error('Invalid container element.');
    }

    const arrowClutter = 30;

    // Caching the container dimensions to minimize layout thrashing
    const width = container.offsetWidth;
    const height = container.offsetHeight;

    // Determine the max value for slides
    let maxValue = vertical ? Math.floor(height * 0.15) - 5 : Math.floor(width * 0.2) - 5;
    maxValue = Math.min(maxValue, 150);

    let slides;
    if (vertical) {
        const slidesMaxWidth = width - arrowClutter * 2 - 10;
        slides = Math.floor(slidesMaxWidth / maxValue);
    } else {
        const slidesMaxHeight = height - arrowClutter * 2 - 20;
        slides = Math.floor(slidesMaxHeight / maxValue);
    }

    return [slides, maxValue];
}

function extract_index(path) {
    const fileName = path.split('/').pop()
    const replaced = fileName.replace(/\D/g, '');
    let num;
    if (replaced !== '') {
        num = Number(replaced) - 1;
    }
    return num
}

function setMorphsButtons(img) {
    for (let i = 0; i < morphed_btns.length; i++) {
        morphed_btns[i].firstChild.src = img + '?' + Math.random();
    }
}

function drawOnCanvas(my_img) {
    context.clearRect(0, 0, canvas.width, canvas.height);
    ref_img.src = my_img;
    ref_img.onload = function () {
        try {
            dsize = new cv.Size(container_center.offsetWidth, container_center.offsetWidth);
            const img = cv.imread(ref_img)
            cv.resize(img, img, dsize, 0, 0, cv.INTER_AREA);
            cv.imshow(canvas, img);
            img.delete();
        } catch (error) {
            console.error('An error occurred in ref_img.onload:', error);
        }
    }
}

function setButtonClick(button, action) {
    button.onclick = action;
}

function clearMatchInterval() {
    if (matchInterval) {
        clearInterval(matchInterval);
        clear_mats()
        reset_bar()
        stopCamera()
        cam_points = {};
        cam_bb = {};
        cam_angles = [];
        cam_expression = [];
        bb_ref_rect = null;
        ref_size = null;
        ref_roi_size = null;
        isMatching = false;
        matchInterval = null;

    }
}

async function init() {
    console.log("init");
    const aspect = container.offsetWidth / container.offsetHeight
    const main_direction = aspect > 1 ? 'horizontal' : 'vertical'

    /* INITIALIZE USER*/
    fetch("/set_user", {
        method: 'POST',
        headers: {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
        body: JSON.stringify({'set': 'user'}),
    })
        .then((res) => {
            if (!res.ok) {
                throw new Error(`HTTP error: ${res.status}`);
            }
            return res.json();
        })
        .then((json) => {
            console.log("USER INITIALIZED")
        })
        .catch((err) => console.error(`Fetch problem: ${err.message}`));

    /* INITIALIZE FACE LANDMARK DETECTOR */
    console.log("initialize detector")
    detector = await faceLandmarksDetection.createDetector(model, detectorConfig);
    console.log("initialized detector")

    /* INITIALIZE UI */
    init_slicks(main_direction);
    updateSlider();
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
                location.reload()
                // drawOnCanvas(start_view);
                // setMorphsButtons(default_morph);
            })
            .catch((err) => console.error(`Fetch problem: ${err.message}`));
        popupWindow.style.display = 'none';
    })
    popupButton.addEventListener('click', () => {
        popupWindow.style.display = 'block';
    });
    sendEmailButton.addEventListener('click', () => {
        const mailToAddress = emailInput.value;
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
    loadJsonFile('../json/ref_images.json')
        .then((json) => {
            face_arr = json;
        });
    console.log('PAINTINGS OBJ INITIALIZED')

    /* INITIALIZE SLICKs BUTTONS AND INTERACTION */
    const reference_btns = container_left.querySelectorAll("div.slick-slide >button");
    morphed_btns = container_right.querySelectorAll("div.slick-slide >button");

    // const btns_indices = [];
    async function handleMainButtonClick(j) {
        selected = j;
        await startCamera();
        let url = `${protocol}//${host}`;
        if (port && selected >= 0) {
            url += `:${port}/${face_arr[selected]['src']}`;

            // url += `/${face_arr[selected]['src']}`;
            ref_img.src = url;
            ref_img.onload = async function () {
                clearMatchInterval()
                morphed = ''
                bb_ref_rect = new cv.Rect
                (
                    face_arr[selected].bb.xMin,
                    face_arr[selected].bb.yMin,
                    face_arr[selected].bb.width,
                    face_arr[selected].bb.height
                );
                ref = cv.imread(ref_img);
                ref_roi = ref.roi(bb_ref_rect);
                ref_size = new cv.Size(ref.cols, ref.rows);
                ref_roi_size = new cv.Size(ref_roi.cols, ref_roi.rows);

                isMatching = true;
                matchInterval = setInterval(match_faces, interval);
                console.log('start interval')
            }
        }
    }

    function handleMorphedButtonClick(j) {

        const m_selected = extract_index(morphed_btns[j].firstChild.src)
        clearMatchInterval()
        ref_img.src = (isNaN(m_selected) === false && j !== extract_index(ref_img.src))
            ? morphed_btns[j].firstChild.src
            : default_view;
        drawOnCanvas(ref_img.src)
        selected = -1
    }

    for (let j = 0; j < reference_btns.length; j++) {
        const r_Btn = reference_btns[j];
        const m_Btn = morphed_btns[j];

        // Main button click action
        setButtonClick(r_Btn, async function () {
            await handleMainButtonClick(j);
        });

        // morphed_btns click action
        setButtonClick(m_Btn, function () {
            handleMorphedButtonClick(j);
        });
    }
    console.log('INTERACTION INITIALIZED')
    body.classList.add('loaded')
}

async function loadJsonFile(filePath) {
    try {
        const response = await fetch(filePath);

        if (!response.ok) {
            throw new Error(`HTTP error: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('Error loading JSON file:', error);
    }
}

/* MANAGE BARS FUNCTIONS */
function reset_bar() {
    [percent_x, percent_y, percent_z].forEach(bar => {
        bar.style.width = bar.innerHTML = '';
    });
}

function update_bar() {
    if (morphed === '') {
        // const selectedFace = face_arr[selected].angles;
        // const camFaceAngles = cam_angles;

        const percentX = 100 - Math.abs(face_arr[selected].angles[0] - cam_angles[0]);
        const percentY = 100 - Math.abs(face_arr[selected].angles[1] - cam_angles[1]);
        const percentZ = 100 - Math.abs(face_arr[selected].angles[2] - cam_angles[2]);

        percent_x.style.width = percentX + '%';
        percent_x.innerHTML = percentX + '%';

        percent_y.style.width = percentY + '%';
        percent_y.innerHTML = percentY + '%';

        percent_z.style.width = percentZ + '%';
        percent_z.innerHTML = percentZ + '%';
    }
}

/* FACE DATA CALCULATIONS */
async function calc_lmrks(image) {
    // console.log('inside calc_lmrks')
    // await detector.reset();
    if (detector) {
        await detector.reset();
    } else{
        detector = await faceLandmarksDetection.createDetector(model, detectorConfig);
    }
    try {
        // let faces = []
        cam_points = {};
        cam_bb = {};
        cam_angles = [];
        cam_expression = [];

        const faces = await detector.estimateFaces(image, {flipHorizontal: false});
        // console.log(faces.length)
        if (faces.length >> 0) {
            const keypoints = faces[0].keypoints;
            cam_points = processKeyPoints(keypoints);
            cam_bb = processBoundingBox(faces[0].box);
            calculateAngles()
            cam_expression = check_expression(cam_points["lmrk13"], cam_points["lmrk14"], cam_points["lmrk33"], cam_points["lmrk78"],
                cam_points["lmrk133"], cam_points["lmrk145"], cam_points["lmrk159"], cam_points["lmrk263"], cam_points["lmrk308"], cam_points["lmrk362"],
                cam_points["lmrk374"], cam_points["lmrk386"]);
        } else {
            no_landmarks()
            // console.log('Cannot calculate landmarks of image')
        }
        // console.log("ended calc_lmrks")
    } catch (error) {
        console.error('Error loading or processing image:', error);
        // Handle the error appropriately, e.g., by returning an error status or rethrowing it.
        throw error;
    }
}

function processKeyPoints(keypoints) {
    const result = {};

    for (let land = 0; land < keypoints.length; land++) {
        if (full_indices_set.has(land)) {
            const x = Math.round(keypoints[land].x);
            const y = Math.round(keypoints[land].y);
            result[`lmrk${land}`] = [x, y];

            if ([10, 133, 152, 226, 362, 446].includes(land)) {
                const z = Math.round(keypoints[land].z);
                result[`lmrk${land}`] = [x, y, z];
            }
        }
    }
    return result;
}

function processBoundingBox(box) {
    return {
        xMin: Math.round(box.xMin),
        xMax: Math.round(box.xMax),
        yMin: Math.round(box.yMin),
        yMax: Math.round(box.yMax),
        width: Math.round(box.width),
        height: Math.round(box.height),
        center: [
            box.xMin + Math.round(box.width / 2),
            box.yMin + Math.round(box.height / 2),
        ],
    };
}

function calculateAngles() {
    if (cam_points["lmrk10"] && cam_points["lmrk133"] && cam_points["lmrk152"] && cam_points["lmrk226"] && cam_points["lmrk362"] && cam_points["lmrk446"]) {
        cam_angles = getHeadAngles(cam_points["lmrk10"], cam_points["lmrk133"], cam_points["lmrk152"], cam_points["lmrk226"], cam_points["lmrk362"], cam_points["lmrk446"]);
    }
}

function getHeadAngles(lmrk10, lmrk133, lmrk152, lmrk226, lmrk362, lmrk446) {
    // V: 10, 152; H: 226, 446
    const faceVerticalCentralPoint = [
        0,
        (lmrk10[1] + lmrk152[1]) * 0.5,
        (lmrk10[2] + lmrk152[2]) * 0.5,
    ];
    const verticalAdjacent = lmrk10[2] - faceVerticalCentralPoint[2];
    const verticalOpposite = lmrk10[1] - faceVerticalCentralPoint[1];
    const verticalHypotenuse = l2Norm([verticalAdjacent, verticalOpposite,]);
    const verticalCos = verticalAdjacent / verticalHypotenuse;

    const faceHorizontalCentralPoint = [
        (lmrk226[0] + lmrk446[0]) * 0.5,
        0,
        (lmrk226[2] + lmrk446[2]) * 0.5,
    ];
    const horizontalAdjacent = lmrk226[2] - faceHorizontalCentralPoint[2];
    const horizontalOpposite = lmrk226[0] - faceHorizontalCentralPoint[0];
    const horizontalHypotenuse = l2Norm([horizontalAdjacent, horizontalOpposite,]);
    const horizontalCos = horizontalAdjacent / horizontalHypotenuse;

    const first = Math.acos(verticalCos) * (180 / Math.PI) - 10;
    const second = Math.acos(horizontalCos) * (180 / Math.PI);
    const third = Math.atan2(lmrk362[1] - lmrk133[1], lmrk362[0] - lmrk133[0]) * (180 / Math.PI)

    return [Math.round(first), Math.round(second), -Math.round(third)];
}

function l2Norm(vector) {
    return Math.sqrt(vector.reduce((acc, val) => acc + val * val, 0));
}

function draw_lines(img, arr) {
    const point1 = new cv.Point(0, 0);
    const point2 = new cv.Point(0, 0);
    const lastPoint = new cv.Point(0, 0);
    const secondLastPoint = new cv.Point(0, 0);
    for (let i = 0; i < arr.length - 1; i++) {
        point1.x = cam_points[`lmrk${arr[i]}`][0] - cam_bb.xMin;
        point1.y = cam_points[`lmrk${arr[i]}`][1] - cam_bb.yMin

        point2.x = cam_points[`lmrk${arr[i + 1]}`][0] - cam_bb.xMin;
        point2.y = cam_points[`lmrk${arr[i + 1]}`][1] - cam_bb.yMin;

        cv.line(img, point1, point2, line_color, 1);
    }
    if (arr.length % 2 === 1) {
        lastPoint.x = cam_points[`lmrk${arr[arr.length - 1]}`][0] - cam_bb.xMin;
        lastPoint.y = cam_points[`lmrk${arr[arr.length - 1]}`][1] - cam_bb.yMin;

        secondLastPoint.x = cam_points[`lmrk${arr[arr.length - 2]}`][0] - cam_bb.xMin;
        secondLastPoint.y = cam_points[`lmrk${arr[arr.length - 2]}`][1] - cam_bb.yMin

        cv.line(img, lastPoint, secondLastPoint, line_color, 1);
    }
}

async function draw_mask_on_ref() {
    if (cam_bb.x === undefined && cam_bb.y === undefined && cam_bb.width === undefined && cam_bb.height === undefined) {
        // console.log('0) cam_bb', cam_bb, ' memory: ', formatCVMemoryUsage());
        return;
    }
    try {
        // const bb_cam_rect = new cv.Rect(cam_bb.xMin, cam_bb.yMin, cam_bb.width, cam_bb.height);
        // console.log('1) bb_cam_rect', bb_cam_rect, ' memory: ', formatCVMemoryUsage())
        // Create ghost source and draw mask on it
        if (cam_bb.xMin > 0 && cam_bb.yMin > 0 && cam_bb.xMin + cam_bb.width < camera_width && cam_bb.yMin + cam_bb.height < camera_height) {
            // console.log('1.1')
            const ghost_source = new cv.Mat.zeros(cam_bb.height, cam_bb.width, cv.CV_8UC3);
            // console.log('1.2', ' memory: ', formatCVMemoryUsage())
            for (const arr of ghost_mask_array) {
                draw_lines(ghost_source, arr);
            }
            cv.flip(ghost_source, ghost_source, 1);
            cv.cvtColor(ghost_source, ghost_source, cv.COLOR_RGB2RGBA, 0);
            cv.resize(ghost_source, ghost_source, ref_roi_size, 0, 0, cv.INTER_LINEAR);
            const sum = new cv.Mat()
            cv.add(ref_roi, ghost_source, sum);
            cv.add(sum, ghost_source, sum);
            releaseMat(ghost_source)
            releaseMat(result)
            result = new cv.Mat();
            ref.copyTo(result);
            sum.copyTo(result.roi(bb_ref_rect));
            releaseMat(sum)
            cv.resize(result, result, dsize, 0, 0, cv.INTER_AREA);
            context.clearRect(0, 0, canvas.width, canvas.height)
            cv.imshow(canvas, result)
            console.log('8) exit function : ', formatCVMemoryUsage())
        } else {
            console.log('face out of frame')
            no_landmarks()
        }
    } catch (error) {
        console.log("matchInterval exists")
        clearMatchInterval()
        // selected = -1;
        const file = "/images/Thumbs/default_view.jpg"
        const path = protocol + '//' + host + ':' + port + file;
        drawOnCanvas(path);
        // const img = new Image();
        // let errorMat;
        // img.onload = function () {
        //     errorMat = cv.imread(img);
        //     cv.imshow(canvas, errorMat);
        // }
        // img.src = path;
        // releaseMat(errorMat)
        // }

        console.error('An error occurred in draw_mask_on_ref:', error);

    }
}

function no_landmarks() {
    if (!result || result.size.width <= 0 || result.size.height <= 0 || result.isDeleted()) {
        console.log('no result')
        result = new cv.Mat.zeros(canvas.height, canvas.width, cv.CV_8UC3);
    }
    const grayResult = new cv.Mat();
    cv.cvtColor(result, grayResult, cv.COLOR_RGBA2GRAY, 0);
    const text = "No face detected"
    // console.log('grayResult memory: ', formatCVMemoryUsage())
    // Calculate the position to center the text
    const where = new cv.Point(((grayResult.cols - text.length * 15) / 2), ((grayResult.rows + 30) / 2));
    cv.putText(
        grayResult,
        text,
        where, // Position of the text
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        line_color,
        2
    );
    // Display the grayed out result with the text
    cv.imshow(canvas, grayResult);
    releaseMat(grayResult)
}

/* MATCHING FUNCTIONS */
async function match_faces() {
    if (!isMatching) {
        return;
    }
    if (selected >= 0 && video.width > 0 && video.height > 0) {
        try {
            await calc_lmrks(video);
            if (Object.keys(cam_points).length === 0 || Object.keys(cam_bb).length === 0 || cam_angles.length === 0) {
                // console.log("No landmarks computed.")
                return;
            }
            update_bar();
            await draw_mask_on_ref();
            await check_and_swap();
        } catch (error) {
            console.error('An error occurred in match_faces:', error);
            clear_mats()
            // init_mats()
        }
    }
}

async function check_and_swap() {
    const delta = 5;
    const [angle1_cam, angle2_cam, angle3_cam] = cam_angles;
    const [angle1_ref, angle2_ref, angle3_ref] = face_arr[selected].angles;
    if (
        angle1_cam >= angle1_ref - delta &&
        angle1_cam <= angle1_ref + delta &&
        angle2_cam >= angle2_ref - delta &&
        angle2_cam <= angle2_ref + delta &&
        angle3_cam >= angle3_ref - delta / 2 &&
        angle3_cam <= angle3_ref + delta / 2
    ) {
        console.log('match1')

        // console.log('cam_expression', cam_expression, 'face_arr[selected].expression', face_arr[selected].expression)
        if (cam_expression.toString() === face_arr[selected].expression.toString()) {
            console.log('match2')
            isMatching = false;
            clearInterval(matchInterval);
            // cam_angles = [];
            // cam_expression = [];
            morphed = ''
            // Draw image on canvas
            ctx.drawImage(video, 0, 0, camera_width, camera_height);
            // Create data URL
            const data_url = cam_canvas.toDataURL('image/jpeg', 0.5);
            // Clear canvas
            ctx.clearRect(0, 0, camera_width, camera_height);
            try {
                const objs = {selected, c_face: data_url};

                const data_json = JSON.stringify(objs);
                // Send data to server

                const res = await fetch('/morph', {
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

                morphed = `${url_base}${url_port}${sub_path}`;
                // Update button image and canvas
                const id = extract_index(rel_path);
                morphed_btns[id].firstChild.src = morphed + '?' + Math.random();
                drawOnCanvas(morphed + '?' + Math.random());
                // reset_bar();
                // stopCamera()
                selected = -1;
                // clearMatchInterval()
                reset_bar()
                stopCamera()
                clear_mats()
                cam_points = {};
                cam_bb = {};
                cam_angles = [];
                cam_expression = [];
                bb_ref_rect = null;
                ref_size = null;
                ref_roi_size = null;
                // isMatching = false;
                matchInterval = null;
            } catch (err) {
                console.error(`Fetch problem: ${err.message}`);
            }


        }
    }
}


function check_expression(lmrk13, lmrk14, lmrk33, lmrk78,
                          lmrk133, lmrk145, lmrk159, lmrk263,
                          lmrk308, lmrk362, lmrk374, lmrk386) {
    try {
        let l_e, r_e, lips

        function calc_division(p1, p2, p3, p4) {
            const p4_p3 = ((p4[0] - p3[0]) ** 2 + (p4[1] - p3[1]) ** 2) ** 0.5
            const p2_p1 = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
            return p4_p3 / p2_p1
        }

        // l_eye
        const l_division = calc_division(lmrk362, lmrk263, lmrk386, lmrk374)
        l_division <= 0.1 ? l_e = 'closed' : l_e = 'opened';
        // r_eye
        const r_division = calc_division(lmrk33, lmrk133, lmrk159, lmrk145)
        r_division <= 0.1 ? r_e = 'closed' : r_e = 'opened';
        // Mouth
        const lips_division = calc_division(lmrk78, lmrk308, lmrk13, lmrk14)
        lips_division < 0.1 ? lips = 'closed' : lips_division > 0.5 ? lips = 'full opened' : lips = 'opened';
        return [l_e, r_e, lips]
    } catch (error) {
        console.error("An error occurred in check_expression:", error);
        // Handle the error or rethrow it if needed
        throw error; // Rethrow the error to propagate it further if necessary
    }
}


async function resize_all() {
    updateSlider()
    ref_img.src = ref_img.src
    if (selected >= 0) {
        await draw_mask_on_ref()
    }
}


function accessCamera() {
    console.log("accessCamera")
    return navigator.mediaDevices
        .getUserMedia({
            video: {facingMode: 'user', width: camera_width, height: camera_height},
            audio: false,
        })
        .catch(function (error) {
            console.log("Something went wrong!", error)
            throw error;
        });
}

function startCamera() {
    return accessCamera()
        .then((stream) => {

            video.srcObject = stream;
        })
        .catch((error) => {
            console.error("Error starting the camera:", error);
        });
}

function stopCamera() {
    if (video.srcObject) {
        const tracks = video.srcObject.getTracks();
        tracks.forEach((track) => {
            track.stop();
        });
        video.srcObject = null;
    }
    if (detector) {
        detector.dispose(); // Example: Release resources associated with the detector
        detector = null; // Set the detector to null to indicate it's no longer available

    }
}

function opencvIsReady() {
    console.log('OPENCV.JS READY')
    init().then(r => {
        console.log('ALL IS INITIALIZED')
    });
}

// function init_mats() {
//     ghost_source = new cv.Mat();
//     sum = new cv.Mat();
// }

function releaseMat(mat) {
    if (mat && !mat.isDeleted()) {
        mat.delete();
        mat = null;
    }
}

function clear_mats() {
    releaseMat(ref)
    releaseMat(ref_roi)
    releaseMat(result)
}

window.addEventListener('resize', resize_all);
window.addEventListener('orientationchange', resize_all);
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
                    console.log('Folder deleted successfully in onbeforeunload');
                } else {
                    console.error('Error deleting folder');
                }
            }).catch(error => {
                console.error(error);
            });
        }
    }, 0);
});

// await
opencvIsReady()


function formatCVMemoryUsage() {
    // Get memory usage from OpenCVJS
    let memoryInfo = cv.getMemory();

    // Convert memory usage to MB
    let memoryInMB = memoryInfo / (1024 * 1024);

    // Format the memory usage with two decimal places
    let formattedMemory = memoryInMB.toFixed(2) + " MB";

    return formattedMemory;
}