
const ref_img = document.getElementById("ref_img");
const video = document.getElementById("webcam");
const middle = document.getElementById("middle");
const canvas = document.getElementById("canvas");
const output1 = document.getElementById("output1");
const output2 = document.getElementById("output2");
const output3 = document.getElementById("output3");
const body = document.body
let context, cam_face,  detector, selected, cam_mat;
let faces_arr=[]
const model = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;
const detectorConfig = {
    runtime: 'mediapipe', // or 'tfjs'
    solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh',
    min_tracking_confidence: 0.2
}
let detect_interval
function canvas_size(value){
    if (value <= 600) {
        canvas.width = value;
        canvas.height = value;
    }
    else {
        canvas.width = 600;
        canvas.height = 600;
    }
}
canvas_size(middle.offsetWidth)
context = canvas.getContext("2d");
const leftSlick = $("#left");
const rightSlick = $("#right")

if (leftSlick.length) {
    leftSlick.slick({
        dots: false,
        infinite: false,
        speed: 300,
        focusOnSelect: false,
        slidesToShow: 3.5,
        slidesToScroll: 2,
        vertical: true,
        // arrows: false,
        draggable: true,
        asNavFor: '.asnav1Class',
        prevArrow: "<button type='button' class='ref_btn'><img src='/images/Thumbs/arrow_up.png' alt='UP'></button>",
        nextArrow: "<button type='button' class='ref_btn'><img src='/images/Thumbs/arrow_down.png' alt='DOWN'></button>"

    });
    rightSlick.slick({
        dots: false,
        infinite: true,
        speed: 300,
        focusOnSelect: false,
        slidesToShow: 3.5,
        slidesToScroll: 2,
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

function path_adjusted(url) {
    if (!/^\w+\:/i.test(url)) {
        url = url.replace(/^(\.?\/?)([\w\@])/, "$1js/$2")
    }
    return url
}

async function init() {
    let btns, all_btns;
    let all_btns_indices = [];

    detector = await faceLandmarksDetection.createDetector(model, detectorConfig);
    fetch(path_adjusted('../TRIANGULATION2.json')).then(response => response.json()).then(data => {TRIANGULATION=data});
    fetch(path_adjusted('../TRIANGULATION2inverted.json')).then(response => response.json()).then(data => {TRIANGULATION_MIRROR=data});
    let container = document.querySelector("#left");
    btns = container.querySelectorAll("div.select_ch.slick-slide:not(.slick-cloned) >button");
    all_btns = container.querySelectorAll("div.slick-slide >button");
    const prop = 'data-slick-index'

    all_btns.forEach(function (item, index, arr){
        all_btns_indices.push(arr[index].parentNode.getAttribute(prop))
    })

    const img_length = btns.length
    for (let imm_num= 0; imm_num<img_length; imm_num++){
        let f = new Face(imm_num, btns[imm_num].firstChild);
        const values = await calc_lmrks(f.image, f.which)
        f.lmrks = values[0]
        f.points = values[1];
        f.n_points = f.normalize_array(values[1])
        f.bb = values[2];
        f.angles = values[3];
        f.hull = convex_hull(values[1], values[2]);
        f.expression = check_expression(values[1])
        faces_arr.push(f)
    }
    for (let j = 0; j<all_btns.length; j++) {
        all_btns[j].onclick = function() {
            let _this = this;
            slide_id = all_btns_indices[j]
            faces_arr.forEach(function (item, index, arr){
                if (_this.firstChild.src === arr[index].src){
                    selected = index
                    ref_img.src = arr[index].src;

                    ref_img.onload = function(){

                        detect_interval = setInterval(detectFaces, 100);
                        // drawHull(ref_img, selected)
                        // draw_landmarks(ref_img.width, ref_img.height,canvas,context,item.lmrks)
                    }
                }
                else {
                    // console.log('btn' + this + ' not matching')
                }
            })
        }
    }
}
function Face(which, image) {
    this.which = which;
    this.image = image;
    this.src = image.src;
    this.w = image.naturalWidth;
    this.h = image.naturalHeight;
    this.points = [];
    this.bb = [];
    this.hull = [];
    this.n_points = []
    this.expression = []

    this.normalize_array= function (pix_points) {
        let n_array = [...pix_points];
        for (let p in n_array){
            n_array[p] = [Math.round(n_array[p][0])/this.w, Math.round(n_array[p][1])/this.h]
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
    }
    else {
        return upperHull.concat(lowerHull);
    }
}
function POINT_COMPARATOR(a, b) {
    // console.log(a +' ' +b)
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
function convex_hull(points, boundingbox){
    let p1, p2, p3, p4, new_arr
    const min_x = boundingbox.xMin
    const max_x = boundingbox.xMax
    const min_y = boundingbox.yMin
    const max_y = boundingbox.yMax
    for (let pt in points){
        if (points[pt][0]===min_x){ p1 = points[pt] }
        else if (points[pt][0]===max_x){ p2 = points[pt] }
        else if (points[pt][1]===min_y){ p3 = points[pt] }
        else if (points[pt][1]===max_y){ p4 = points[pt] }
    }
    let poly = [p1, p2, p3, p4]
    new_arr = [...points]

    let _new_arr = [...new_arr]
    for (let pt in new_arr){
        if (pointInPolygon(poly, new_arr[pt]) === true){
            _new_arr = arrayRemove(_new_arr, new_arr[pt])
        }
    }
    for (let el in poly){
        if (!(poly[el] in _new_arr)){_new_arr.push(poly[el])}
    }
    _new_arr.sort(POINT_COMPARATOR);
    let hull = calcHull(_new_arr)
    let _hull = [];
    for (let h in hull){
        for (let p in points){
            if (hull[h] === points[p]){
                _hull.push(p)
            }
        }
    }
    return _hull
}
function arrayRemove(arr, value) {
    return arr.filter(function(ele){
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

async function getHeadAngles(keypoints) {
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
    const first = Math.acos(verticalCos) * (180 / Math.PI);
    const second = Math.acos(horizontalCos)  * (180 / Math.PI);
    let pleft = keypoints[133]
    let pright = keypoints[362]
    let third = Math.atan2(pright[1]-pleft[1], pright[0]-pleft[0])*(180/Math.PI)
    return [first, second, third];
}

function l2Norm(vec) {
    let norm = 0;
    for (const v of vec) {
        norm += v * v;
    }
    return Math.sqrt(norm);
}

async function calc_lmrks(image, which) {
    // console.log('calcolo FACE '+ which)
    let landmarks = []
    let points_2d = []
    let bb ={}
    let angles
    let faces = await detector.estimateFaces(image);
    if (faces.length>>0){
        const keypoints = faces[0].keypoints;
        for (let land=0; land< keypoints.length; land++){
            let x = Math.round(keypoints[land].x);
            let y = Math.round(keypoints[land].y);
            let z = Math.round(keypoints[land].z);
            landmarks.push([x, y, z])
            points_2d.push([x,y])
        }
        bb.xMin = Math.round(faces[0].box.xMin)
        bb.xMax = Math.round(faces[0].box.xMax)
        bb.yMin = Math.round(faces[0].box.yMin)
        bb.yMax = Math.round(faces[0].box.yMax)
        bb.width = Math.round(faces[0].box.width)
        bb.height = Math.round(faces[0].box.height)
        bb.center = [bb.xMin + Math.round(bb.width/2),bb.yMin + Math.round(bb.height/2)]


        if (landmarks !== []) {
            // console.log('not none')
            angles = await getHeadAngles(landmarks)
        }
    }
    else{
        console.log('non ce la faccio a calcolare le faces dell\'imm ' + which)
        // console.log(this.src)
    }
    return [landmarks, points_2d, bb, angles]
}

function distance(a,b) {
    return Math.sqrt(Math.pow(a[0]-b[0],2) + Math.pow(a[1]-b[1],2))
}

function draw_landmarks( w, h,canvas_where,ctx,land_array){

    ctx.globalAlpha = 0.5
    ctx.save()
    ctx.clearRect(0,0,w,h)
    ctx.fillStyle = 'black'
    ctx.fillRect(0,0,w,h)
    ctx.fillStyle = '#32EEDB';
    ctx.strokeStyle = '#32EEDB';
    ctx.lineWidth = 0.5;
    for (let tris_indices in TRIANGULATION){
        const points_indices = [TRIANGULATION[tris_indices][0],TRIANGULATION[tris_indices][1],
            TRIANGULATION[tris_indices][2]]
        const points = [land_array[points_indices[0]], land_array[points_indices[1]],
            land_array[points_indices[2]]]
        const region = new Path2D();
        region.moveTo(points[0][0], points[0][1]);
        for (let i = 1; i < 3; i++) {
            const point = points[i];
            region.lineTo(point[0], point[1]);
        }
        region.closePath();

        ctx.stroke(region);
    }
}

function drawHull(image, idx){
    const w = canvas.width
    const h = canvas.width
    const this_picture = faces_arr[idx]
    const points_indices = this_picture.hull;
    let hull_points = [];
    for (let el in points_indices){
        let id = points_indices[el];
        let new_point = new cv.Point(this_picture.n_points[id][0]*w,this_picture.n_points[id][1]*h)
        hull_points.push(new_point)
    }
    const mat = cv.imread(image)
    let larger_mat = new cv.Mat()
    const dsize = new cv.Size(w, h)
    cv.resize(mat, larger_mat, dsize, 0, 0, cv.INTER_LINEAR);
    for (let lin = 0;lin < hull_points.length-1; lin++){
        let p1  = hull_points[lin];
        let p2  = hull_points[lin+1];
        cv.line(larger_mat, p1, p2, [0, 255, 0, 255], 1)
    }
    cv.imshow(canvas, larger_mat);
    mat.delete(); larger_mat.delete()

}

function hull_mask(){
    let hull_points = [];
    for (let el in cam_face.hull){
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
    cv.drawContours(convexHullMat, hulls, 0, [255,255,255,0], -1, 8);
    hull.delete(); hulls.delete()
    return convexHullMat
}
function mask_from_array(points, w, h){

    let mask_from_array_mat = cv.Mat.zeros(w, h, cv.CV_8UC3);
    let points_to_mat = cv.matFromArray(points.length, 1, cv.CV_32SC2, points.flat())
    let fake_vector = new cv.MatVector();
    fake_vector.push_back(points_to_mat)
    cv.drawContours(mask_from_array_mat, fake_vector, 0, [255,255,255,0], -1, 8)
    fake_vector.delete(); points_to_mat.delete()
    // cv.imshow(canvas,mask_from_array_mat)
    return mask_from_array_mat
}

function draw_mask_on_ref(){
    cam_mat=new cv.Mat()
    const ref_bb = faces_arr[selected].bb
    const bb_cam_rect = new cv.Rect(cam_face.bb.xMin,cam_face.bb.yMin,cam_face.bb.width,cam_face.bb.height)
    const bb_ref_rect = new cv.Rect(ref_bb.xMin,ref_bb.yMin,ref_bb.width,ref_bb.height);
    let cap = new cv.VideoCapture(video);
    let cam_source = new cv.Mat(video.height, video.width,  cv.CV_8UC4)
    cap.read(cam_source)
    let temp_mask = new cv.Mat(video.height, video.width,  cv.CV_8UC1)
    let cam_roi= cam_source.roi(bb_cam_rect)

    const ref = cv.imread(ref_img)

    const ref_roi = ref.roi(bb_ref_rect);
    // cam_source.copyTo(cam_mat, temp_mask)
    cam_mat =  cam_source.clone()
    // console.log(cam_mat)
    cam_source.delete(); temp_mask.delete()
    // sizes
    const dsize = new cv.Size(cam_roi.cols, cam_roi.rows)
    const dsize_back = new cv.Size(ref_roi.cols, ref_roi.rows)
    canvas_size(middle.offsetWidth)
    const last_size =  new cv.Size(canvas.width, canvas.height)

    cv.GaussianBlur(cam_roi, cam_roi, new cv.Size(3, 3), 0, 0, cv.BORDER_DEFAULT)

    let cam_roi_gray = new cv.Mat()
    cv.cvtColor(cam_roi,cam_roi_gray, cv.COLOR_RGBA2GRAY, 0)

    let cam_laplacian = laplacian(cam_roi, cam_roi.cols, cam_roi.rows)
    cam_roi.delete();
    
    let cam_sobel = new cv.Mat()
    cv.Sobel(cam_roi_gray, cam_sobel, cv.CV_8U, 1, 0, 3, 1, 0, cv.BORDER_DEFAULT)
    cam_roi_gray.delete();
    
    let mask = new  cv.Mat()
    cv.addWeighted(cam_sobel, .75, cam_laplacian, .25, 0, mask);
    cam_laplacian.delete(); cam_sobel.delete();

    let new_mask = new cv.Mat()
    ///// calcolo convex hull mask
    let convexHullMat = hull_mask()
    let cam_mask = convexHullMat.roi(bb_cam_rect);
    convexHullMat.delete();
    cv.cvtColor(cam_mask,cam_mask, cv.COLOR_RGBA2GRAY, 0)
    cv.bitwise_and(mask, mask, new_mask, cam_mask)
    cam_mask.delete(); mask.delete();
    
    cv.flip(new_mask, new_mask,1)

    let color_new_mask = new cv.Mat()
    cv.cvtColor(new_mask, color_new_mask, cv.COLOR_GRAY2RGB, 0)
    new_mask.delete();
    
    // const pass = color_new_mask.clone();
    // cv.bilateralFilter(pass, color_new_mask, 5, 75, 75, cv.BORDER_DEFAULT);
    // pass.delete();
    cv.cvtColor(color_new_mask, color_new_mask, cv.COLOR_RGB2RGBA, 0)
    
    cv.resize(ref_roi, ref_roi, dsize, 0, 0, cv.INTER_LINEAR )
    let sum = new cv.Mat();
    cv.add(ref_roi,color_new_mask,sum)
    ref_roi.delete();
    cv.add(sum,color_new_mask,sum)
    color_new_mask.delete();
    cv.resize(sum, sum, dsize_back, 0, 0, cv.INTER_LINEAR )

    let dst = ref.clone();
    ref.delete();
    for (let i = 0; i < sum.rows; i++) {
        for (let j = 0; j < sum.cols; j++) {
            dst.ucharPtr(i+ref_bb.yMin, j+ref_bb.xMin)[0] = sum.ucharPtr(i, j)[0];
            dst.ucharPtr(i+ref_bb.yMin, j+ref_bb.xMin)[1] = sum.ucharPtr(i, j)[1];
            dst.ucharPtr(i+ref_bb.yMin, j+ref_bb.xMin)[2] = sum.ucharPtr(i, j)[2]
        }
    }
    sum.delete();

    cv.resize(dst, dst, last_size, 0, 0, cv.INTER_LINEAR )
    cv.imshow(canvas, dst);

    dst.delete()
}

function laplacian(src, width, height) {
    const dstC1 = new cv.Mat(height, width, cv.CV_8UC1)
    let mat = new cv.Mat(height, width, cv.CV_8UC1);
    cv.cvtColor(src, mat, cv.COLOR_RGB2GRAY);
    cv.Laplacian(mat, dstC1, cv.CV_8U, 5, 1, 0, cv.BORDER_DEFAULT);
    mat.delete();
    return dstC1;
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
            console.log("Something went wrong!");
        });
};

function check_expression(landmarks){
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
    const lips_division= calc_division(landmarks[78], landmarks[308], landmarks[13], landmarks[14])
    lips_division < 0.15 ? lips = 'closed' : 0.15 <= lips_division < 0.4 ? lips = 'opened' : lips = 'full opened';

    return [l_e, r_e, lips]
}

function match(angles_cam, angles_ref){
    draw_mask_on_ref()
    const delta = 8;
    if ((angles_cam[0] >= angles_ref[0] - delta && angles_cam[0] <= angles_ref[0] + delta) &&
    (angles_cam[1] >= 180- angles_ref[1] - delta && angles_cam[1] <= 180-angles_ref[1] + delta) &&
    (angles_cam[2] >= -angles_ref[2] - delta/2 && angles_cam[2] <= -angles_ref[2] + delta/2)){
        console.log('match1')
        if (cam_face.expression.toString() === faces_arr[selected].expression.toString()) {
            console.log('match2')
            // cv.imshow(canvas, cam_mat)

            swap_face()
            clearInterval(detect_interval)
            selected = -1
        }
    }
    const dsize = new cv.Size(cam_mat.cols/10, cam_mat.rows/10)
    // if (cam_mat){
    //     cv.resize(cam_mat, cam_mat, dsize, 0, 0, cv.INTER_LINEAR )
    //     cv.imshow(output, cam_mat)
    // }

    cam_mat = null
}

function swap_face() {
    const mc = match_c(ref_img, cam_mat)
    let tris1 = [], tris2 = [];
    for (let i in TRIANGULATION){
        let tri1 = [], tri2 = [];
        const tris_indices = TRIANGULATION[i]
        const tris_indices_mirror = TRIANGULATION_MIRROR[i]
        for (let j = 0; j < 3; j++) {
            tri1.push([cam_face.points[tris_indices_mirror[j]][0],cam_face.points[tris_indices_mirror[j]][1]])
            tri2.push([faces_arr[selected].points[tris_indices[j]][0],faces_arr[selected].points[tris_indices[j]][1]])
        }
        tris1.push(tri1)
        tris2.push(tri2)
    }
    let target_mat = cv.imread(ref_img)
    let cam_hist = calc_Hist(cam_mat)
    let ref_hist = calc_Hist(target_mat)
    // cam_mat = equalize_hist(cam_mat)
    for (let i in tris1) {
        let warp = warp_triangle(cam_mat, target_mat, tris1[i], tris2[i])

        target_mat = warp.clone()

        warp.delete()
    }
    cv.imshow(canvas, target_mat)
    // cam_mat = null
    tris1 = []
    tris2 = []
    target_mat.delete()

}

function warp_triangle(source_mat, target, t1, t2 ){
    let bb1 = {
        minX: Math.min(t1[0][0], t1[1][0], t1[2][0]),
        minY: Math.min(t1[0][1], t1[1][1], t1[2][1]),
        maxX: Math.max(t1[0][0], t1[1][0], t1[2][0]),
        maxY: Math.max(t1[0][1], t1[1][1], t1[2][1]),
    }
    bb1.w = bb1.maxX - bb1.minX
    bb1.h = bb1.maxY - bb1.minY

    let bb2 = {
        minX: Math.min(t2[0][0], t2[1][0], t2[2][0]),
        minY: Math.min(t2[0][1], t2[1][1], t2[2][1]),
        maxX: Math.max(t2[0][0], t2[1][0], t2[2][0]),
        maxY: Math.max(t2[0][1], t2[1][1], t2[2][1]),
    }
    bb2.w = bb2.maxX - bb2.minX
    bb2.h = bb2.maxY - bb2.minY
    if ( bb1.w >> 0 || bb1.h >> 0 || bb2.w >> 0 || bb2.h >> 0){
        const r1 = new cv.Rect(bb1.minX, bb1.minY, bb1.w, bb1.h);
        const r2 = new cv.Rect(bb2.minX, bb2.minY, bb2.w, bb2.h);

        //normalize points , inverting x values in t1
        let n_t1 = [], n_t2 = []
        for (let i = 0; i < 3; i++){
            let new_point1 = [Math.round((1-(t1[i][0]- bb1.minX)/(bb1.maxX-bb1.minX)) * bb2.w), Math.round(((t1[i][1] - bb1.minY)/(bb1.maxY-bb1.minY)) * bb2.h)]
            n_t1.push(new_point1)
            let new_point2 = [Math.round(((t2[i][0]- bb2.minX)/(bb2.maxX-bb2.minX)) * bb2.w), Math.round(((t2[i][1]- bb2.minY)/(bb2.maxY-bb2.minY)) * bb2.h)]
            n_t2.push(new_point2)
        }
        let srcTri = cv.matFromArray(3, 1, cv.CV_32FC2, n_t1.flat());
        let dstTri = cv.matFromArray(3, 1, cv.CV_32FC2, n_t2.flat());

        let M = cv.getAffineTransform(srcTri, dstTri);
        srcTri.delete(); dstTri.delete();
        M.data64F[2] += (M.data64F[0]+M.data64F[1]-1)/2
        M.data64F[5] += (M.data64F[3]+M.data64F[4]-1)/2

        let roi_t2 = target.roi(r2)
        let roi_t2_mask = new cv.Mat(roi_t2.rows, roi_t2.cols,  cv.CV_8UC1)
        let warped = new cv.Mat()
        roi_t2.copyTo(warped, roi_t2_mask)
        roi_t2_mask.delete()
        let roi_t1 = source_mat.roi(r1)
        let dsize = new cv.Size(bb2.w, bb2.h);
        cv.resize(roi_t1, roi_t1, dsize, 0, 0, cv.INTER_AREA )
        cv.flip(roi_t1,roi_t1, 1)
        cv.warpAffine(roi_t1, warped, M, dsize, cv.INTER_AREA, cv.BORDER_DEFAULT ,[0, 0, 0, 0]);// new cv.Scalar()
        M.delete(); roi_t1.delete();
        let mask_t2 = mask_from_array(t2, target.cols, target.rows)
        let roi_mask_t2 = mask_t2.roi(r2);
        mask_t2.delete();

        //convert to 1 channel gray
        cv.cvtColor(roi_mask_t2,roi_mask_t2, cv.COLOR_RGBA2GRAY, 0);
        // inverse Roi mask t2
        let roi_mask_t2_inv = new cv.Mat()
        cv.bitwise_not(roi_mask_t2,roi_mask_t2_inv)
        let mix = mask_img( roi_t2, warped, roi_mask_t2,roi_mask_t2_inv)
        roi_mask_t2.delete(); roi_mask_t2_inv.delete(); warped.delete(); roi_t2.delete();
        let dst = target.clone();
        for (let i = 0; i < mix.rows; i++) {
            for (let j = 0; j < mix.cols; j++) {
                dst.ucharPtr(i+bb2.minY, j+bb2.minX)[0] = mix.ucharPtr(i, j)[0];
                dst.ucharPtr(i+bb2.minY, j+bb2.minX)[1] = mix.ucharPtr(i, j)[1];
                dst.ucharPtr(i+bb2.minY, j+bb2.minX)[2] = mix.ucharPtr(i, j)[2]
            }
        }
        mix.delete()
        return dst
    }
    else{ console.log(bb1.w +' '+ bb1.h +' '+ bb2.w +' '+ bb2.h )}
    // cv.fillConvexPoly(mask, new_triangle,t2,3, new cv.Scalar(1,1,1,1) )
}
function mask_img(bg_img, fg_img, mask, mask_inv) {
    let imgBg = new cv.Mat(bg_img.cols, bg_img.rows, cv.CV_8UC4);
    let imgFg = new cv.Mat(bg_img.cols, bg_img.rows, cv.CV_8UC4);
    let sum = new cv.Mat();
    // Black-out the area of logo in ROI
    cv.bitwise_and(bg_img, bg_img, imgBg, mask_inv);

    // Take only region of logo from logo image
    cv.bitwise_and(fg_img, fg_img, imgFg, mask);

    // Put logo in ROI and modify the main image
    cv.add(imgBg, imgFg, sum);
    imgBg.delete();imgFg.delete();//bg_img.delete();fg_img.delete();mask_inv.delete();
    return sum
}
async function detectFaces(){
    cam_face = new Face('cam', video)
    cam_face.w=video.width;
    cam_face.h=video.height;

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
            // console.log(bb)
            if (selected>=0) {

                match(cam_face.angles, faces_arr[selected].angles)
            }
        })
        .catch((err) => {
            console.log(err);
        });
}

function resizeCanvas() {
    canvas_size(middle.offsetWidth)
    // output.width =  middle.offsetWidth ;
    // output.height = middle.offsetWidth ;
    /**
     * Your drawings need to be inside this function otherwise they will be reset when
     * you resize the browser window and the canvas goes will be cleared.
     */
    if (selected>=0){
        draw_mask_on_ref()
        // drawHull(ref_img, selected);
    }
}
init();
accessCamera();


window.addEventListener('resize', resizeCanvas, false);
video.addEventListener('loadeddata',function() {
    body.classList.add('loaded')

    // setInterval(detectFaces, 100);
    })
function calc_Hist(src0){
    let src = src0.clone()
    let rgbaPlanes = new cv.MatVector();
// Split the Mat
    cv.split(src, rgbaPlanes);
// Get R channel
    let R = rgbaPlanes.get(0);
    let G = rgbaPlanes.get(1);
    let B = rgbaPlanes.get(2);
    let A = rgbaPlanes.get(3);
    const channels = [R, G, B, A]
    const colors = [new cv.Scalar(255, 0, 0),new cv.Scalar(0, 255, 0),new cv.Scalar(0, 0, 255)]
    let rVec = new cv.MatVector();
    let gVec = new cv.MatVector();
    let bVec = new cv.MatVector();
    const vectors = [rVec, gVec, bVec]
    let hists = []
    let outputs = [output1,output2,output3]
    for (let i=0; i<3; i++) {
        vectors[i].push_back(channels[i]);

        let accumulate = false;
        let channel = [0];
        let histSize = [256];
        let ranges = [0, 255];
        let hist = new cv.Mat();
        let mask = new cv.Mat();
        let color = colors[i];
        let scale = 2;
        // You can try more different parameters
        cv.calcHist(vectors[i], channel, mask, hist, histSize, ranges, accumulate);
        let hist_ = hist.clone()
        let result = cv.minMaxLoc(hist, mask);
        let max = result.maxVal;
        let dst = new cv.Mat.zeros(src.rows/5, histSize[0] * scale,
            cv.CV_8UC3);
        // draw histogram
        for (let i = 0; i < histSize[0]; i++) {
            let binVal = hist.data32F[i] * src.rows/5 / max;
            let point1 = new cv.Point(i * scale, src.rows/5 - 1);
            let point2 = new cv.Point((i + 1) * scale - 1, src.rows/5 - binVal);
            cv.rectangle(dst, point1, point2, color, cv.FILLED);
        }
        cv.imshow(outputs[i], dst);
        hists.push(hist_)
        mask.delete(); hist.delete(); dst.delete()
    }

    // console.log(hists)
    rVec.delete(); gVec.delete(); bVec.delete(); rgbaPlanes.delete(); src.delete()
    return hists
}
function calc_Hist_old(src0){
    let src = src0.clone()
    cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
    let rVec = new cv.MatVector();
    rVec.push_back(src);
    let accumulate = false;
    let channels = [0];
    let histSize = [256];
    let ranges = [0, 255];
    let hist = new cv.Mat();
    let mask = new cv.Mat();
    let color = new cv.Scalar(255, 255, 255);
    let scale = 2;
    // You can try more different parameters
    cv.calcHist(rVec, channels, mask, hist, histSize, ranges, accumulate);
    let result = cv.minMaxLoc(hist, mask);
    let max = result.maxVal;
    let dst = new cv.Mat.zeros(src.rows, histSize[0] * scale,
        cv.CV_8UC3);
    // draw histogram
    for (let i = 0; i < histSize[0]; i++) {
        let binVal = hist.data32F[i] * src.rows / max;
        let point1 = new cv.Point(i * scale, src.rows - 1);
        let point2 = new cv.Point((i + 1) * scale - 1, src.rows - binVal);
        cv.rectangle(dst, point1, point2, color, cv.FILLED);
    }
    cv.imshow('output', dst);
    src.delete(); dst.delete(); rVec.delete(); mask.delete(); hist.delete();
}
function equalize_hist(src){
    // let imgElement = document.getElementById("ImgViewImage"); // img element with ImgViewImage id
    // let src = cv.imread(imgElement);
    console.log(src.channels())
    let dst = new cv.Mat();
    let hsvPlanes = new cv.MatVector();
    let mergedPlanes = new cv.MatVector();
    // cv.cvtColor(src, src, cv.COLOR_RGB2HSV, 0);
    cv.split(src, hsvPlanes);
    let H = hsvPlanes.get(0);
    let S = hsvPlanes.get(1);
    let V = hsvPlanes.get(2);
    cv.equalizeHist(V, V);
    mergedPlanes.push_back(H);
    mergedPlanes.push_back(S);
    mergedPlanes.push_back(V);
    cv.merge(mergedPlanes, src);
    // cv.cvtColor(src, dst, cv.COLOR_HSVRGBA, 0);
    // cv.imshow("canvasOutput", dst); // canavas element with canvasOutput id
    src.delete();
    // dst.delete();
    hsvPlanes.delete();
    mergedPlanes.delete();
    return dst
}


function python_match_hist(){
    function find_value_target(val, target_arr) {
        let key = np.where(target_arr == val)[0]

        if (len(key) == 0) {
            key = find_value_target(val + 1, target_arr)

            if (len(key) == 0)
            {
                key = find_value_target(val - 1, target_arr)
            }}
        let vvv = key[0]
        return vvv
    }


    function match_histogram(inp_img, hist_input, e_hist_input, e_hist_target, _print=True) {
        // map from e_inp_hist to 'target_hist

        let en_img = np.zeros_like(inp_img)
        let tran_hist = np.zeros_like(e_hist_input)
        for (let i in range(len(e_hist_input))) {
            tran_hist[i] = find_value_target(val = e_hist_input[i], target_arr = e_hist_target)
        }
        // print_histogram(tran_hist, name = "trans_hist_", title = "Transferred Histogram")
        // enhance image as well:'

        for (let x_pixel in (inp_img.cols)) {
            for (let y_pixel in inp_img.rows){
                let pixel_val = Math.round(inp_img[x_pixel][y_pixel])
                en_img[x_pixel][y_pixel] = tran_hist[pixel_val]
            }
        }
        ''
        'creating new histogram'
        ''
        hist_img, _ = generate_histogram(en_img, print = False, index = 3)
        print_img(img = en_img, histo_new = hist_img, histo_old = hist_input, index = str(3), L = L)
    }
}

////////////////////////////////////////

