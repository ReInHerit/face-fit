//const spawn = require('child_process').spawn;
const ref_img = document.getElementById("ref_img");
const video = document.getElementById("webcam");
const middle = document.getElementById("middle");
const canvas = document.getElementById("canvas");
const perc_x = document.getElementById("pb_x");
const perc_y = document.getElementById("pb_y");
const perc_z = document.getElementById("pb_z");
const ref_x = document.getElementById("ref_x");
const ref_y = document.getElementById("ref_y");
const ref_z = document.getElementById("ref_z");
const cam_x = document.getElementById("cam_x");
const cam_y = document.getElementById("cam_y");
const cam_z = document.getElementById("cam_z");
const container = document.getElementsByClassName("container")[0]
let container_left = document.querySelector("#left");
let container_right = document.querySelector("#right")
let container_center = document.querySelector("#demos")
const cam_canvas = document.getElementById('cam_canvas');
const ctx = cam_canvas.getContext('2d');
const body = document.body
let context, cam_face, detector, selected, cam_mat;
let face_arr = []
const model = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;
const detectorConfig = {
    runtime: 'mediapipe',
    solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh',
    min_tracking_confidence: 0.2
}
const default_view = '../images/Thumbs/default_view.jpg'
const start_view = 'http://localhost:8000/images/Thumbs/start_view.jpg'

let detect_interval
let morphed = ''
let m_all_btns
const leftSlick = $("#left");
const rightSlick = $("#right")

if (leftSlick.length) {
    leftSlick.slick({
        dots: false,
        infinite: false,
        speed: 300,
        focusOnSelect: false,
        slidesToShow: 3.5,
        slidesToScroll: 3,
        vertical: true,
        // arrows: false,
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
        // arrows: false,
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

function canvas_size(value) {

    if (value <= 600) {

        canvas.width = value;
        canvas.height = value;
    } else {
        canvas.width = 600;
        canvas.height = 600;
    }
}
function window_size(){
    console.log(container.offsetWidth)
    console.log(container.offsetHeight)
    if(container.offsetWidth <= 800){

        canvas.height = container.offsetWidth * 0.6;
        canvas.width = canvas.height;
//        container.style.gridTemplate = '[header-left] "view" 1fr [header-right] [main-left] "ref_btns"  30px [main-right] [footer-left] "morph_btns" 30px [footer-right]/ 120px 1fr;'
        container.style.gridTemplateAreas=   '"view" "ref_btns" "morph_btns"';
        container.style.gridTemplateColumns = '1fr';
        container.style.gridTemplateRows = '60% 10% 10%';
        leftSlick.slick['vertical'] = false;
        rightSlick.slick['vertical'] = false;
    }
    else {
//    container.style.gridTemplate ='[header-left] "ref_btns view morph_btns" 30px [header-right]/ 120px 1fr;'
        container.style.gridTemplateAreas=  'ref_btns view morph_btns';

        container.style.gridTemplateColumns = '20% 60% 20%';
        container.style.gridTemplateRows = '100%';
        leftSlick.slick['vertical'] = true;
        rightSlick.slick['vertical'] = true;
        canvas.width = 600;
        canvas.height = 600;
    }
}

//canvas_size(middle.offsetWidth)
window_size()
context = canvas.getContext("2d");

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

async function init() {
    let btns, m_btns, all_btns;
    let all_btns_indices = [];
    ref_img.src = start_view
    ref_img.onload = function(){
        load_on_ref_canvas(ref_img)
    }

    detector = await faceLandmarksDetection.createDetector(model, detectorConfig);
    fetch(path_adjusted('../TRIANGULATION2.json')).then(response => response.json()).then(data => {
        TRIANGULATION = data
    });
    fetch(path_adjusted('../TRIANGULATION2inverted.json')).then(response => response.json()).then(data => {
        TRIANGULATION_MIRROR = data
    });

    btns = container_left.querySelectorAll("div.select_ch.slick-slide:not(.slick-cloned) >button");
    all_btns = container_left.querySelectorAll("div.slick-slide >button");
    m_btns = container_right.querySelectorAll("div.select_ch.slick-slide:not(.slick-cloned) >button");
    m_all_btns = container_right.querySelectorAll("div.slick-slide >button");
    const prop = 'data-slick-index'
    all_btns.forEach(function (item, index, arr) {
        all_btns_indices.push(arr[index].parentNode.getAttribute(prop))
    })
    paintings = {'start': 'paintings'}
    await fetch("/init", {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
                    body: JSON.stringify(paintings)
                })
                .then((res) => {
                if (!res.ok) {
                    throw new Error(`HTTP error: ${res.status}`);
                }
                console.log(res.json)
                return res.json();
            })
                .then((json) => {
                console.log(json)
                face_arr = json.body
            })
    const img_length = btns.length

    for (let j = 0; j < all_btns.length; j++) {
        all_btns[j].onclick = function () {
            let _this = this;
            slide_id = all_btns_indices[j]
            selected = extract_index(_this.firstChild.src)
            source = 'http://localhost:8000/' + face_arr[selected]['src']
            console.log(source)
            ref_img.src = source;

            ref_img.onload = function () {
                console.log(selected)
                if (selected !== -1){
                morphed = ''
                detect_interval = setInterval(detectFaces, 1000/30);
                console.log('start interval')
                }
            }
        }

        m_all_btns[j].onclick = function () {
            let _this = this;
            slide_id = all_btns_indices[j]
            m_selected = extract_index(_this.firstChild.src)
            if (m_selected !== '' && slide_id != extract_index(ref_img.src)){
                ref_img.src = _this.firstChild.src

            }
//            else if (selected >= 0) {
//                clearInterval(detect_interval)
//                ref_img.src = _this.firstChild.src
//            }
            else {
                ref_img.src = default_view
            }

        }
    }
}

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

function convex_hull(points, boundingbox) {
    let p1, p2, p3, p4, new_arr
    const min_x = boundingbox.xMin
    const max_x = boundingbox.xMax
    const min_y = boundingbox.yMin
    const max_y = boundingbox.yMax
    for (let pt in points) {
        if (points[pt][0] === min_x) {
            p1 = points[pt]
        } else if (points[pt][0] === max_x) {
            p2 = points[pt]
        } else if (points[pt][1] === min_y) {
            p3 = points[pt]
        } else if (points[pt][1] === max_y) {
            p4 = points[pt]
        }
    }
    let poly = [p1, p2, p3, p4]
    new_arr = [...points]

    let _new_arr = [...new_arr]
    for (let pt in new_arr) {
        if (pointInPolygon(poly, new_arr[pt]) === true) {
            _new_arr = arrayRemove(_new_arr, new_arr[pt])
        }
    }
    for (let el in poly) {
        if (!(poly[el] in _new_arr)) {
            _new_arr.push(poly[el])
        }
    }
    _new_arr.sort(point_comparator);
    let hull = calcHull(_new_arr)
    let _hull = [];
    for (let h in hull) {
        for (let p in points) {
            if (hull[h] === points[p]) {
                _hull.push(p)
            }
        }
    }
    return _hull
}

function arrayRemove(arr, value) {
    return arr.filter(function (ele) {
        return ele !== value;
    });
}

function pointInPolygon(polygon, point) {
    //A point is in a polygon if a line from the point to infinity crosses the polygon an odd number of times
    let odd = false;
    //For each edge (In this case for each point of the polygon and the previous one)
    for (let i = 0, j = polygon.length - 1; i < polygon.length; i++) {
        //If a line from the point into infinity crosses this edge
        if (((polygon[i][1] > point[1]) !== (polygon[j][1] > point[1])) // One point needs to be above, one below our y coordinate
            // ...and the edge doesn't cross our Y coordinate before our x coordinate (but between our x coordinate and infinity)
            && (point[0] < ((polygon[j][0] - polygon[i][0]) * (point[1] - polygon[i][1]) / (polygon[j][1] - polygon[i][1]) + polygon[i][0]))) {
            // Invert odd
            odd = !odd;
        }
        j = i;
    }
    //If the number of crossings was odd, the point is in the polygon
    return odd;
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

function l2Norm(vec) {
    let norm = 0;
    for (const v of vec) {
        norm += v * v;
    }
    return Math.sqrt(norm);
}

function normalize(value, bounds){
    new_value = bounds['desired']['lower'] + (value - bounds['actual']['lower']) *
    (bounds['desired']['upper'] - bounds['desired']['lower']) /
    (bounds['actual']['upper'] - bounds['actual']['lower'])
    return new_value
}

async function calc_lmrks(image, which) {
    // console.log('calcolo FACE '+ which)
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
//        console.log(landmarks)
        bb.xMin = Math.round(faces[0].box.xMin)
        bb.xMax = Math.round(faces[0].box.xMax)
        bb.yMin = Math.round(faces[0].box.yMin)
        bb.yMax = Math.round(faces[0].box.yMax)
        bb.width = Math.round(faces[0].box.width)
        bb.height = Math.round(faces[0].box.height)
        bb.center = [bb.xMin + Math.round(bb.width / 2), bb.yMin + Math.round(bb.height / 2)]


        if (landmarks !== []) {
            // console.log('not none')
            angles = await getHeadAngles(landmarks, which)
        }
    } else {
        console.log('non ce la faccio a calcolare le faces dell\'imm ' + which)
        // console.log(this.src)
    }
    return [landmarks, points_2d, bb, angles]
}


function hull_mask() {
    let hull_points = [];
    for (let el in cam_face.hull) {
        let id = cam_face.hull[el];
        let new_point = [cam_face.n_points[id][0] * cam_face.w, cam_face.n_points[id][1] * cam_face.h]
        hull_points.push(new_point)
    }
    let convexHullMat = cv.Mat.zeros(cam_face.w, cam_face.h, cv.CV_8UC3);
    let hull = cv.matFromArray(hull_points.length, 1, cv.CV_32SC2, hull_points.flat());
    // make a fake hulls vector
    let hulls = new cv.MatVector();
    // add the recently created hull
    hulls.push_back(hull);
    // test drawing it
    cv.drawContours(convexHullMat, hulls, 0, [255, 255, 255, 0], -1, 8);
    hull.delete();
    hulls.delete()
    return convexHullMat
}


function draw_mask_on_ref() {
    cam_mat = new cv.Mat()
    const ref_bb = face_arr[selected].bb
    console.log(face_arr[selected])
    const bb_cam_rect = new cv.Rect(cam_face.bb.xMin, cam_face.bb.yMin, cam_face.bb.width, cam_face.bb.height)
    const bb_ref_rect = new cv.Rect(ref_bb.xMin, ref_bb.yMin, ref_bb.width, ref_bb.height);
    let cap = new cv.VideoCapture(video);
    let cam_source = new cv.Mat(video.height, video.width, cv.CV_8UC4)
    cap.read(cam_source)
    let temp_mask = new cv.Mat(video.height, video.width, cv.CV_8UC1)
    let cam_roi = cam_source.roi(bb_cam_rect)
    const ref = cv.imread(ref_img)
    const ref_roi = ref.roi(bb_ref_rect);
    cam_mat = cam_source.clone()
    cam_source.delete();
    temp_mask.delete()
    // sizes
    const dsize = new cv.Size(cam_roi.cols, cam_roi.rows)
    const dsize_back = new cv.Size(ref_roi.cols, ref_roi.rows)
//    canvas_size(middle.offsetWidth)
    window_size()
    const last_size = new cv.Size(canvas.width, canvas.height)

    cv.GaussianBlur(cam_roi, cam_roi, new cv.Size(1, 1), 0, 0, cv.BORDER_DEFAULT)

    let cam_roi_gray = new cv.Mat()
    cv.cvtColor(cam_roi, cam_roi_gray, cv.COLOR_RGBA2GRAY, 0)

    let cam_laplacian = laplacian(cam_roi, cam_roi.cols, cam_roi.rows)
    cam_roi.delete();
    // let cam_canny = new cv.Mat()
    // cv.Canny(cam_roi_gray, cam_canny, 1, 30, 3, false)
    // cv.cvtColor(cam_canny,cam_canny, cv.COLOR_GRAY2RGBA, 0)
    let cam_sobel = new cv.Mat()
    let abs_cam_sobel = new cv.Mat();
    cv.Sobel(cam_roi_gray, cam_sobel, cv.CV_8U, 1, 0, 3, 1, 0, cv.BORDER_DEFAULT)
    cv.Sobel(cam_roi_gray, abs_cam_sobel, cv.CV_64F, 1, 0, 3, 1, 0, cv.BORDER_DEFAULT)
    cv.convertScaleAbs(abs_cam_sobel, abs_cam_sobel, 1, 0);
    cam_roi_gray.delete();
    cam_sobel.delete();

    let mask = new cv.Mat()
    cv.addWeighted(cam_laplacian, 1, abs_cam_sobel, 1, 0, mask);
    cam_laplacian.delete();
    abs_cam_sobel.delete();

    let new_mask = new cv.Mat()
    ///// calcolo convex hull mask
    let convexHullMat = hull_mask()
    let cam_mask = convexHullMat.roi(bb_cam_rect);
    convexHullMat.delete();
    cv.cvtColor(cam_mask, cam_mask, cv.COLOR_RGBA2GRAY, 0)
    cv.bitwise_and(mask, mask, new_mask, cam_mask)
    cam_mask.delete();
    mask.delete();

    cv.flip(new_mask, new_mask, 1)

    let color_new_mask = new cv.Mat()
    cv.cvtColor(new_mask, color_new_mask, cv.COLOR_GRAY2RGB, 0)
    new_mask.delete();

    // const pass = color_new_mask.clone();
    // cv.bilateralFilter(pass, color_new_mask, 5, 75, 75, cv.BORDER_DEFAULT);
    // pass.delete();
    cv.cvtColor(color_new_mask, color_new_mask, cv.COLOR_RGB2RGBA, 0)

    cv.resize(ref_roi, ref_roi, dsize, 0, 0, cv.INTER_LINEAR)
    let sum = new cv.Mat();
    cv.add(ref_roi, color_new_mask, sum)
    ref_roi.delete();
    cv.add(sum, color_new_mask, sum)
    color_new_mask.delete();
    cv.resize(sum, sum, dsize_back, 0, 0, cv.INTER_LINEAR)

    let dst = ref.clone();
    ref.delete();
    for (let i = 0; i < sum.rows; i++) {
        for (let j = 0; j < sum.cols; j++) {
            dst.ucharPtr(i + ref_bb.yMin, j + ref_bb.xMin)[0] = sum.ucharPtr(i, j)[0];
            dst.ucharPtr(i + ref_bb.yMin, j + ref_bb.xMin)[1] = sum.ucharPtr(i, j)[1];
            dst.ucharPtr(i + ref_bb.yMin, j + ref_bb.xMin)[2] = sum.ucharPtr(i, j)[2]
        }
    }
    sum.delete();

    cv.resize(dst, dst, last_size, 0, 0, cv.INTER_LINEAR)
    cv.imshow(canvas, dst);
    // cv.imshow(canvas, abs_cam_sobel);

    dst.delete()
}

function laplacian(src, height, width) {
    const dstC1 = new cv.Mat(height, width, cv.CV_8UC1)
    let mat = new cv.Mat(height, width, cv.CV_8UC1);
    cv.cvtColor(src, mat, cv.COLOR_RGB2GRAY);
    cv.Laplacian(mat, dstC1, cv.CV_8U, 1, 1, 0, cv.BORDER_DEFAULT);
    mat.delete();
    return dstC1;
}

function check_expression(landmarks) {
    function calc_division(land_id1, land_id2, land_id3, land_id4) {
        const p1 = [land_id1[0], land_id1[1], 0]
        const p2 = [land_id2[0], land_id2[1], 0]
        const p3 = [land_id3[0], land_id3[1], 0]
        const p4 = [land_id4[0], land_id4[1], 0]
        const p4_p3 = ((p4[0] - p3[0]) ** 2 + (p4[1] - p3[1]) ** 2 + (p4[2] - p3[2]) ** 2) ** 0.5
        const p2_p1 = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2) ** 0.5
        // const division = p4_p3 / p2_p1
        // center = find_center(np.array([p1, p2, p3, p4]))
        return p4_p3 / p2_p1//, center
    }

    // l_eye
    let l_e, r_e, lips
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

async function match(angles_cam, angles_ref) {
    const delta = 8;
    console.log(angles_cam[0], angles_ref[0])
    console.log(angles_cam[1], angles_ref[1])
    console.log(angles_cam[2], angles_ref[2])
    if ((angles_cam[0] >= angles_ref[0] - delta && angles_cam[0] <= angles_ref[0] + delta) &&
        (angles_cam[1] >= angles_ref[1] - delta && angles_cam[1] <= angles_ref[1] + delta) &&
        (angles_cam[2] >= angles_ref[2] - delta / 2 && angles_cam[2] <= angles_ref[2] + delta / 2)) {
        console.log('match1')
        if (cam_face.expression.toString() === face_arr[selected].expression.toString()) {
            console.log('match2')
            // setTimeout(() => {
            morphed = ''
            ctx.drawImage(video, 0, 0, cam_face.w, cam_face.h);
            let data_url = cam_canvas.toDataURL('image/jpeg', 0.5);
            ctx.clearRect(0, 0, cam_face.w, cam_face.h);
            cam_face.image = data_url

            let objs = {'selected': selected, 'c_face': cam_face.image}//'p_face': face_arr[selected],
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
                console.log(res.json)
                return res.json();
            })
            .then((json) => {
                let rel_path = json.relative_path
                morphed = rel_path + '?' + Math.random()
                id = extract_index(rel_path)
//                let dsize = new cv.Size(middle.offsetWidth, middle.offsetWidth);
                m_all_btns[id].firstChild.src = morphed
                ref_img.src = morphed
                ref_img.onload = function () {
                    load_on_ref_canvas(ref_img)
//                    morph = cv.imread(ref_img)
//                    console.log(morphed, morph)
//                    cv.resize(morph, morph, dsize, 0, 0, cv.INTER_AREA);
//                    cv.imshow(canvas, morph);
                    console.log('loaded')
                    clearInterval(detect_interval)

                    }
//                ref_img.src = ref_img.src

                
                reset_bar()
                selected = -1


            })
            .catch((err) => console.error(`Fetch problem: ${err.message}`));

        }
    }
    console.log(selected, morphed)

    cam_mat = null
}
function load_on_ref_canvas(image) {
    let dsize = new cv.Size(middle.offsetWidth, middle.offsetWidth);
//    m_all_btns[id].firstChild.src = morphed
//    ref_img.src = morphed
//    ref_img.onload = function () {
    img = cv.imread(image)
//    console.log(morphed, morph)
    cv.resize(img, img, dsize, 0, 0, cv.INTER_AREA);
    cv.imshow(canvas, img);
    }
function reset_bar(){
    ref_x.innerHTML = '-'
    ref_y.innerHTML = '-'
    ref_z.innerHTML = '-'
    cam_x.innerHTML = '-'
    cam_y.innerHTML = '-'
    cam_z.innerHTML = '-'
    perc_x.style.width = 1 + '%';
    perc_x.innerHTML = ''
    perc_y.style.width = 1 + '%';;
    perc_y.innerHTML = ''
    perc_z.style.width = 1 + '%';;
    perc_z.innerHTML = ''
}
function update_bar() {

    let percent_x = 100 - Math.abs(face_arr[selected].angles[0] - cam_face.angles[0])
    let percent_y = 100 - Math.abs(face_arr[selected].angles[1] - cam_face.angles[1])
    let percent_z = 100 - Math.abs(face_arr[selected].angles[2] - cam_face.angles[2])
//    var identity = setInterval(scene, 1000/30);
//    function scene() {
        if (morphed != '') {
//            clearInterval(identity);
        } else {
            ref_x.innerHTML = face_arr[selected].angles[0]
            ref_y.innerHTML = face_arr[selected].angles[1]
            ref_z.innerHTML = face_arr[selected].angles[2]
            cam_x.innerHTML = cam_face.angles[0]
            cam_y.innerHTML = cam_face.angles[1]
            cam_z.innerHTML = cam_face.angles[2]
            perc_x.style.width = percent_x + '%';
            perc_x.innerHTML = percent_x + '%'
            perc_y.style.width = percent_y + '%';
            perc_y.innerHTML = percent_y + '%'
            perc_z.style.width = percent_z + '%';
            perc_z.innerHTML = percent_z + '%'
//        }
    }
}

async function detectFaces() {
//accessCamera();
if (selected >= 0) {

    console.log('detect cycle:', selected)
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
            match(cam_face.angles, face_arr[selected].angles)

        })
        .catch((err) => {
            console.log(err);
        });
}}

function resizeCanvas() {
    console.log('resize')
//    canvas_size(middle.offsetWidth)
    window_size()
    /**
     * Your drawings need to be inside this function otherwise they will be reset when
     * you resize the browser window and the canvas goes will be cleared.
     */
    ref_img.src = ref_img.src
    if (selected >= 0) {
        draw_mask_on_ref()
        // drawHull(ref_img, selected);
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
//            video.style.webkitTransform = "scaleX(1)";
//            video.style.transform       = "scaleX(1)";
        })
        .catch(function (error) {
            console.log("Something went wrong!", error);
        });
};
function opencvIsReady(){
    init();
    accessCamera();
}


window.addEventListener('resize', resizeCanvas, false);
video.addEventListener('loadeddata', function () {
    body.classList.add('loaded')
})



