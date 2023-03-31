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
const morphs_path = '../morphs'
const default_view = '../images/Thumbs/default_view.jpg'
const default_morph = '../images/Thumbs/morph_thumb.jpg'
const send_logo = '../images/Thumbs/send.png'
const start_view = '../images/Thumbs/start_view.jpg'

/* UI FUNCTIONS*/
const leftSlick = $("#ref_btns");
const rightSlick = $("#morph_btns")

function set_slick(orientation, slides, arrows){
    let prev_string = '<button type="button" class="ref_btn"><img src="/images/Thumbs/arrow_' + arrows[0] +'.png" alt="PREV"></button>';
    let next_string = '<button type="button" class="ref_btn"><img src="/images/Thumbs/arrow_' + arrows[1] + '.png" alt="NEXT"></button>';

    leftSlick.slick('slickSetOption','slidesToShow', slides);
    rightSlick.slick('slickSetOption','slidesToShow', slides);
    leftSlick.slick('slickSetOption','vertical', orientation);
    rightSlick.slick('slickSetOption','vertical', orientation);
    leftSlick.slick('slickSetOption','prevArrow', prev_string)
    leftSlick.slick('slickSetOption','nextArrow', next_string)
    rightSlick.slick('slickSetOption','prevArrow', prev_string)
    rightSlick.slick('slickSetOption','nextArrow', next_string)
    leftSlick.slick( 'refresh' );
    rightSlick.slick( 'refresh' );
}

function init_slicks(){
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

function window_size(){
    let orientation, arrows, areas, columns, rows;
    const width =  container.offsetWidth
    let slides = (width <= 700 && width >= 600) ? 6 : (width < 600 && width >= 450) ? 5 : (width < 450 ) ? 4 : 3.5
    if(width <= 700){
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

    }
    else {
        areas = '"ref_btns main_view main_view morph_btns" "ref_btns main_view main_view morph_btns" "ref_btns main_view main_view morph_btns"';
        columns = '20% 30% 30% 20%';
        rows = '40% 40% 20%';
        container_center.style.maxHeight = Math.round(container.offsetHeight - hints.offsetHeight - 40) + 'px';
        container_center.style.width = Math.round(container.offsetWidth * 0.6 -20) + 'px';
        container_left.style.display = container_right.style.display = 'block';
        arrows = ['up', 'down'];
        orientation = true
    }
    canvas.style.maxHeight = hints.style.maxWidth = container_center.style.maxHeight
    container.style.gridTemplateAreas = areas
    container.style.gridTemplateColumns = columns;
    container.style.gridTemplateRows = rows;

    container_center.style.maxWidth = Math.round(container.offsetWidth - 20) + 'px';
    set_slick(orientation ,slides, arrows)
}

function path_adjusted(url) {
    if (!/^\w+:/i.test(url)) {
        url = url.replace(/^(\.?\/?)([\w@])/, "$1js/$2")
    }
    return url
}

function extract_index(path){
    const fileName = path.split('/').pop()
    const replaced = fileName.replace(/\D/g, ''); // 👉️ '123'
    let num;
    if (replaced !== '') {
      num = Number(replaced)-1; // 👉️ 123
    }
    return num
}

function setMorphsButtons(img){
    for (let i = 0; i < m_all_btns.length; i++) {
        m_all_btns[i].firstChild.src = img;
  }
}

function drawOnCanvas(my_img){
    ref_img.src = my_img;
    ref_img.onload = function(){
        let dsize = new cv.Size(container_center.offsetWidth, container_center.offsetWidth);
        let img = cv.imread(ref_img)
        cv.resize(img, img, dsize, 0, 0, cv.INTER_AREA);
        cv.imshow(canvas, img);
    }
}

function setButtonClick(button, action) {
  button.onclick = action;
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
    popupButton.addEventListener('click', () => {popupWindow.style.display = 'block';});
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
        }
        else {
            if(confirm("Your input is not a valid email address.\n" +
                "Press Cancel to retry or Ok to reset the game!")){

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
            } else{
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
                if (selected !== -1){
                    morphed = ''
                    detect_interval = setInterval(match_faces, 1000/30);
                    console.log('start interval')
                }
            }
        });

        setButtonClick(m_all_btns[j], function () {
            selected = -1
            let _this = this;
            let slide_id = all_btns_indices[j]
            const m_selected = extract_index(_this.firstChild.src)
            if (detect_interval){
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
    let faces = await detector.estimateFaces(image);
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
    let first = Math.acos(verticalCos) *(180 / Math.PI);
    let second = Math.acos(horizontalCos) * (180 / Math.PI);
    if (which === 'cam') {
            first = Math.round(normalize(first, {'actual': {'lower': 55, 'upper': 115}, 'desired': {'lower': 82, 'upper': 95}}))
            second = Math.round(normalize(second, {'actual': {'lower': 50, 'upper': 120}, 'desired': {'lower': 120, 'upper': 69}}))
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

function normalize(value, bounds){
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
function draw_mask_on_ref() {
    const ref_bb = face_arr[selected].bb
    const bb_cam_rect = new cv.Rect(cam_face.bb.xMin, cam_face.bb.yMin, cam_face.bb.width, cam_face.bb.height)
    const bb_ref_rect = new cv.Rect(ref_bb.xMin, ref_bb.yMin, ref_bb.width, ref_bb.height);

    //  cam & ref rois
    let cap = new cv.VideoCapture(video);
    let cam_source = new cv.Mat(video.height, video.width, cv.CV_8UC4)
    cap.read(cam_source)
    let cam_roi = cam_source.roi(bb_cam_rect)
    cam_source.delete();

    const ref = cv.imread(ref_img)
    const ref_roi = ref.roi(bb_ref_rect);

    // sizes
    const dsize = new cv.Size(cam_roi.cols, cam_roi.rows)
    const dsize_back = new cv.Size(ref_roi.cols, ref_roi.rows)
    const last_size = new cv.Size(canvas.width, canvas.height)
    
    // create ghost mask
    cv.GaussianBlur(cam_roi, cam_roi, ...[new cv.Size(1, 1), 0, 0, cv.BORDER_DEFAULT])

    let cam_roi_gray = new cv.Mat()
    cv.cvtColor(cam_roi, cam_roi_gray, cv.COLOR_RGBA2GRAY, 0)

    let cam_laplacian = laplacian(cam_roi, cam_roi.cols, cam_roi.rows)
    cam_roi.delete();

    let cam_sobel = new cv.Mat()
    let abs_cam_sobel = new cv.Mat();
    cv.Sobel(cam_roi_gray, cam_sobel, cv.CV_8U, 1, 0, 3, 1, 0, cv.BORDER_DEFAULT)
    cv.Sobel(cam_roi_gray, abs_cam_sobel, cv.CV_64F, 1, 0, 3, 1, 0, cv.BORDER_DEFAULT)
    cv.convertScaleAbs(abs_cam_sobel, abs_cam_sobel, 1, 0);
    cam_roi_gray.delete(); cam_sobel.delete();

    let mask = new cv.Mat()
    cv.addWeighted(cam_laplacian, 1, abs_cam_sobel, 1, 0, mask);
    cam_laplacian.delete(); abs_cam_sobel.delete();

    let convexHullMat = hull_mask()
    let cam_mask = convexHullMat.roi(bb_cam_rect);
    convexHullMat.delete();
    cv.cvtColor(cam_mask, cam_mask, cv.COLOR_RGBA2GRAY, 0)
    let ghost_mask_gray = new cv.Mat()
    cv.bitwise_and(mask, mask, ghost_mask_gray, cam_mask)
    cam_mask.delete(); mask.delete();

    cv.flip(ghost_mask_gray, ghost_mask_gray, 1)

    let ghost_mask = new cv.Mat()
    cv.cvtColor(ghost_mask_gray, ghost_mask, cv.COLOR_GRAY2RGB, 0)
    ghost_mask_gray.delete();
    cv.cvtColor(ghost_mask, ghost_mask, cv.COLOR_RGB2RGBA, 0)
    
    // apply ghost mask over ref_image
    cv.resize(ref_roi, ref_roi, dsize, 0, 0, cv.INTER_LINEAR);
    let sum = new cv.Mat();
    cv.add(ref_roi, ghost_mask, sum);
    ref_roi.delete();
    cv.add(sum, ghost_mask, sum)
    ghost_mask.delete();
    cv.resize(sum, sum, dsize_back, 0, 0, cv.INTER_LINEAR)

    let dst = ref.clone();
    ref.delete();
    sum.copyTo(dst.roi(bb_ref_rect));
    sum.delete();

    cv.resize(dst, dst, last_size, 0, 0, cv.INTER_LINEAR)
    cv.imshow(canvas, dst);
    dst.delete()
}

function hull_mask() {
    const hull_points = cam_face.hull.map((id) => [ cam_face.n_points[id][0] * cam_face.w,
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

function laplacian(src, height, width) {
    const dstC1 = new cv.Mat(height, width, cv.CV_8UC1)
    let mat = new cv.Mat(height, width, cv.CV_8UC1);
    cv.cvtColor(src, mat, cv.COLOR_RGB2GRAY);
    cv.Laplacian(mat, dstC1, cv.CV_8U, 1, 1, 0, cv.BORDER_DEFAULT);
    mat.delete();
    return dstC1;
}


/* MATCHING FUNCTIONS */
async function match_faces() {
    if (selected >= 0) {
        cam_face = new Face('cam', video)
        cam_face.w = video.width;
        cam_face.h = video.height;

        const cam_promise = calc_lmrks(video, 'cam')
        cam_promise
            .then((value) => {
                cam_face.lmrks = value[0];
                cam_face.points = value[1];
                cam_face.bb = value[2];
                cam_face.angles = value[3];
                cam_face.hull = convex_hull(value[1], value[2]);
                cam_face.n_points = cam_face.normalize_array(value[1])
                cam_face.expression = check_expression(value[1])
                update_bar()

                draw_mask_on_ref()
                check_and_swap(cam_face.angles, face_arr[selected].angles)

            })
            .catch((err) => {
                console.log(err);
            });
    }
}

async function check_and_swap(angles_cam, angles_ref) {
    const delta = 8;
    if ((angles_cam[0] >= angles_ref[0] - delta && angles_cam[0] <= angles_ref[0] + delta) &&
        (angles_cam[1] >= angles_ref[1] - delta && angles_cam[1] <= angles_ref[1] + delta) &&
        (angles_cam[2] >= angles_ref[2] - delta / 2 && angles_cam[2] <= angles_ref[2] + delta / 2)) {
        console.log('match1')
        if (cam_face.expression.toString() === face_arr[selected].expression.toString()) {
            console.log('match2')
            clearInterval(detect_interval)
            morphed = ''
            ctx.drawImage(video, 0, 0, cam_face.w, cam_face.h);
            let data_url = cam_canvas.toDataURL('image/jpeg', 0.5);
            ctx.clearRect(0, 0, cam_face.w, cam_face.h);
            cam_face.image = data_url
            let objs = {'selected': selected, 'c_face': cam_face.image}
            let data_json = JSON.stringify(objs)

            await fetch("/info", {
                method: 'POST',
                headers: {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
                body: data_json
            })
            .then((res) => {
                if (!res.ok) {
                    throw new Error(`HTTP error: ${res.status}`);
                }
                return res.json();
            })
            .then((json) => {
                let rel_path = json.relative_path
                if (port === '') {
                    morphed = protocol + '//' + host + rel_path; //+ '?' + Math.random()
                } else {
                    morphed = protocol + '//' + host + ':' + port + rel_path //+ '?' + Math.random()
                }
                let id = extract_index(rel_path)
                m_all_btns[id].firstChild.src = morphed
                drawOnCanvas(morphed)
                reset_bar()
                selected = -1
            })
            .catch((err) => console.error(`Fetch problem: ${err.message}`));
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
    lips_division < 0.15 ? lips = 'closed' : 0.15 <= lips_division < 0.4 ? lips = 'opened' : lips = 'full opened';
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
            video: {width: 1280, height: 960},
            audio: false,
        })
        .then((stream) => {
            video.srcObject = stream;
        })
        .catch(function (error) {
            console.log("Something went wrong!", error);
        });
};
function opencvIsReady(){
    console.log('OPENCV.JS READY')
    init().then(r => console.log('ALL IS INITIALIZED'));
    accessCamera();
}

window.addEventListener('resize', resize_all, false);
video.addEventListener('loadeddata', function () {
    body.classList.add('loaded')
})



