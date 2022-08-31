
const ref_img = document.getElementById("ref_img");
const video = document.getElementById("webcam");
const middle = document.getElementById("middle");
const canvas = document.getElementById("canvas");
const body = document.body
let context, RAF_timerID, facemesh_drawn, cam_face, faces, container, detector, selected, cam_mat, ref_mat;
let cam_landmarks = [], cam_box = [], btns=[], faces_arr=[], all_btns = [], all_btns_indices = []
const model = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;
const detectorConfig = {
    runtime: 'mediapipe', // or 'tfjs'
    solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh',
}

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
// const border_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
//     152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
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
        prevArrow: "<button type='button' class='ref_btn'><image src='/images/Thumbs/arrow_up.png'></image></button>",
        nextArrow: "<button type='button' class='ref_btn'><image src='/images/Thumbs/arrow_down.png'></image></button>"

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
        prevArrow: "<button type='button' class='ref_btn'><image src='/images/Thumbs/arrow_up.png'></image></button>",
        nextArrow: "<button type='button' class='ref_btn'><image src='/images/Thumbs/arrow_down.png'></image></button>"
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
    detector = await faceLandmarksDetection.createDetector(model, detectorConfig);
    fetch(path_adjusted('../TRIANGULATION2.json')).then(response => response.json()).then(data => {TRIANGULATION=data});
    fetch(path_adjusted('../TRIANGULATION2inverted.json')).then(response => response.json()).then(data => {TRIANGULATION_MIRROR=data});
    container = document.querySelector("#left");
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
        // console.log(which)
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
    // if ((canvas_where.width != w) || (canvas_where.height != h)) {
    //     canvas_where.width  = w
    //     canvas_where.height = h
    // }

    ctx.globalAlpha = 0.5

    ctx.save()

    ctx.clearRect(0,0,w,h)

    ctx.fillStyle = 'black'
    ctx.fillRect(0,0,w,h)

    if (!TRIANGULATION ) { //|| !faces.length
        facemesh_drawn = false
        ctx.restore()
        return
    }
    facemesh_drawn = true
    ctx.fillStyle = '#32EEDB';
    ctx.strokeStyle = '#32EEDB';
    ctx.lineWidth = 0.5;
    for (let tris_indices in TRIANGULATION){
        const points_indices = [TRIANGULATION[tris_indices][0],TRIANGULATION[tris_indices][1],
            TRIANGULATION[tris_indices][2]]
        const points = [land_array[points_indices[0]], land_array[points_indices[1]],
            land_array[points_indices[2]]]
        drawPath(ctx, points, true);
    }
}
function drawPath(ctx, points, closePath) {
    const region = new Path2D();
    region.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i < 3; i++) {
        const point = points[i];
        region.lineTo(point[0], point[1]);
    }

    if (closePath) {
        region.closePath();
    }
    ctx.stroke(region);
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
    mat.delete()
    larger_mat.delete()
}
function cv_draw(webcam, bb){
    // console.log(bb.xMin+ ' ' + bb.yMin+ ' ' + bb.width+ ' ' + bb.height)
    // let rect1,rect2,rect3,rect4
    // let rect
    let rect, dst
    // output.width =  middle.offsetWidth ;
    // output.height = middle.offsetWidth ;

    let src2 = new cv.Mat(webcam.height, webcam.width,  cv.CV_8UC4)
    // let ref = new cv.Mat(ref_img.offsetWidth,ref_img.offsetHeight)
    rect = new cv.Rect(bb.xMin,bb.yMin,bb.width,bb.height)
    // if (bb){
    //     console.log('yes')
    //     rect= new cv.Rect(rect1,rect2,rect3,rect4)
    // }
    // else{
    //     console.log('no')
    //     rect = new cv.Rect(0, 0, 50, 50)
    // }
    // let nuevo = cv.flip(src2,dst, 1)
    // let p01, p02, p03, p04, p05, p06, p07, p08
    // p01 =new cv.Point(bb.xMin, bb.yMin)
    // p02 =new cv.Point(bb.xMin,bb.yMax)
    // p03 =new cv.Point(bb.xMax,bb.yMax)
    // p04 =new cv.Point(bb.xMax,bb.yMin)
    // p05=new cv.Point(webcam.width, webcam.height)
    // p06=new cv.Point(webcam.width,0 )
    // p07=new cv.Point(0, 0)
    // p08=new cv.Point(0, webcam.height)
    // src2=data
    // console.log(dst.cols)
    // dst = src2;

    let cap = new cv.VideoCapture(webcam);
    cap.read(src2)
    dst= src2.roi(rect)
    // cv.line(passage, p01, p02, [255, 0, 0, 255], 1)
    // cv.line(passage, p05, p06, [255, 0, 0, 255], 2)
    // cv.line(passage, p06, p07, [255, 255, 0, 255], 2)
    // cv.line(passage, p07, p08, [0, 255, 0, 255], 2)
    // cv.line(passage, p08, p05, [0, 255, 255, 255], 2)
    // passage = passage.roi(rect)

    // const dsize = new cv.Size(webcam.offsetWidth, webcam.offsetHeight)
    // cv.resize(passage, passage, dsize, 0, 0, cv.INTER_LINEAR);
    // let p1, p2, p3, p4, p5,p6,p7,p8,new_bb_xMin, new_bb_yMin, new_bb_xMax, new_bb_yMax
    // p1=new cv.Point(bb.xMin*webcam.offsetHeight/webcam.height, bb.yMin*webcam.offsetWidth/webcam.width)
    // p2=new cv.Point(bb.xMin*webcam.offsetHeight/webcam.height, bb.yMax*webcam.offsetWidth/webcam.width)
    // p3=new cv.Point(bb.xMax*webcam.offsetHeight/webcam.height, bb.yMax*webcam.offsetWidth/webcam.width)
    // p4=new cv.Point(bb.xMax*webcam.offsetHeight/webcam.height, bb.yMin*webcam.offsetWidth/webcam.width)
    //
    //
    //
    // cv.line(passage, p1, p2, [255, 0, 0, 255], 1)
    // cv.line(passage, p2, p3, [255, 0, 0, 255], 1)
    // cv.line(passage, p3, p4, [255, 0, 0, 255], 1)
    // cv.line(passage, p4, p1, [255, 0, 0, 255], 1)


    // console.log(cap)
    // let crop=new cv.Mat(webcam.height, webcam.width, cv.CV_8UC1);
    // let dsize = new cv.Size(48, 48);
    // cv.cvtColor(dst, gray, cv.COLOR_RGBA2GRAY, 0);
    // cv.imshow(output, dst);
    src.delete();
    dst.delete();

}
function hull_mask(face_obj){
    let hull_points = [];
    for (let el in face_obj.hull){
        let id = face_obj.hull[el];
        let new_point = [face_obj.n_points[id][0]*face_obj.w,face_obj.n_points[id][1]*face_obj.h]
        hull_points.push(new_point)
    }
    let convexHullMat = cv.Mat.zeros(face_obj.w, face_obj.h, cv.CV_8UC3);
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

function draw_mask_on_ref(webcam, cam_obj){
    const ref_bb = faces_arr[selected].bb
    let bb_cam_rect, cam_roi, bb_ref_rect, ref_roi, cam_mask, dst
    // let maskInv = new cv.Mat();
    let mask = new  cv.Mat()
    let new_mask = new  cv.Mat()
    let color_new_mask = new cv.Mat()
    let sum = new cv.Mat();
    let cam_source = new cv.Mat(webcam.height, webcam.width,  cv.CV_8UC4)
    // let cam_mask_inv = new cv.Mat()
    let cam_roi_gray = new cv.Mat()
    let cam_laplacian // = new cv.Mat()
    let cam_sobel = new cv.Mat()
    let ref = cv.imread(ref_img)
    let cap = new cv.VideoCapture(webcam);

    cap.read(cam_source)
    // console.log(cam_source.cols)
    cam_mat = cam_source.clone()
    // cv.flip(cam_mat, cam_mat, 1)
    bb_cam_rect = new cv.Rect(cam_obj.bb.xMin,cam_obj.bb.yMin,cam_obj.bb.width,cam_obj.bb.height)
    bb_ref_rect = new cv.Rect(ref_bb.xMin,ref_bb.yMin,ref_bb.width,ref_bb.height);

    ///// calcolo convex hull mak
    let convexHullMat = hull_mask(cam_obj)

    // ROIs
    cam_mask = convexHullMat.roi(bb_cam_rect)
    cam_roi= cam_source.roi(bb_cam_rect)
    ref_roi = ref.roi(bb_ref_rect);
    cam_source.delete(); convexHullMat.delete();
    // sizes
    const dsize = new cv.Size(cam_roi.cols, cam_roi.rows)
    const dsize_back = new cv.Size(ref_roi.cols, ref_roi.rows)
    // create inverse cam mask
    cv.cvtColor(cam_mask,cam_mask, cv.COLOR_RGBA2GRAY, 0)
    // cv.bitwise_not(cam_mask, cam_mask_inv);

    cv.GaussianBlur(cam_roi, cam_roi, new cv.Size(3, 3), 0, 0, cv.BORDER_DEFAULT)
    cv.cvtColor(cam_roi,cam_roi_gray, cv.COLOR_RGBA2GRAY, 0)

    cam_laplacian = laplacian(cam_roi, cam_roi.cols, cam_roi.rows)

    cv.Sobel(cam_roi_gray, cam_sobel, cv.CV_8U, 1, 0, 3, 1, 0, cv.BORDER_DEFAULT)
    cam_roi_gray.delete();
    cv.addWeighted(cam_sobel, .75, cam_laplacian, .25, 0, mask);

    cam_laplacian.delete(); cam_sobel.delete();

    cv.bitwise_and(mask, mask, new_mask, cam_mask)
    cam_mask.delete();
    cv.flip(new_mask, new_mask,1)
    // cv.bitwise_not(new_mask, maskInv);

    cv.resize(ref_roi, ref_roi, dsize, 0, 0, cv.INTER_LINEAR )

    cv.cvtColor(new_mask, color_new_mask, cv.COLOR_GRAY2RGB, 0)
    new_mask.delete();
    let pass = color_new_mask.clone();
    cv.bilateralFilter(pass, color_new_mask, 5, 75, 75, cv.BORDER_DEFAULT);
    pass.delete();
    cv.cvtColor(color_new_mask, color_new_mask, cv.COLOR_RGB2RGBA, 0)
    cv.add(ref_roi,color_new_mask,sum)
    cv.add(sum,color_new_mask,sum)
    color_new_mask.delete();
    cv.resize(sum, sum, dsize_back, 0, 0, cv.INTER_LINEAR )
    cv.resize(cam_roi, cam_roi, dsize_back, 0, 0, cv.INTER_LINEAR )

    dst = ref.clone();
    for (let i = 0; i < cam_roi.rows; i++) {
        for (let j = 0; j < cam_roi.cols; j++) {
            dst.ucharPtr(i+ref_bb.yMin, j+ref_bb.xMin)[0] = sum.ucharPtr(i, j)[0];
            dst.ucharPtr(i+ref_bb.yMin, j+ref_bb.xMin)[1] = sum.ucharPtr(i, j)[1];
            dst.ucharPtr(i+ref_bb.yMin, j+ref_bb.xMin)[2] = sum.ucharPtr(i, j)[2]
        }
    }
    sum.delete();
    canvas_size(middle.offsetWidth)
    let last_size =  new cv.Size(canvas.width, canvas.height)
    cv.resize(dst, dst, last_size, 0, 0, cv.INTER_LINEAR )
    cv.imshow(canvas, dst);
    ref.delete(); dst.delete(); cam_roi.delete(); ref_roi.delete(); mask.delete();
    // maskInv.delete();  cam_mask_inv.delete();


}
function nestedPointsArrayToMat(points){
    return cv.matFromArray(points.length, 1, cv.CV_32SC2, points.flat());
}


function laplacian(src, width, height) {
    const dstC1 = new cv.Mat(height, width, cv.CV_8UC1)
    const mat = new cv.Mat(height, width, cv.CV_8UC1);
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
    const delta = 10;
    if ((angles_cam[0]>= angles_ref[0] - delta && angles_cam[0] <= angles_ref[0] + delta) &&
    (angles_cam[1]>= 180- angles_ref[1] - delta && angles_cam[1]<= 180-angles_ref[1] + delta) &&
    (angles_cam[2]>= -angles_ref[2] - delta/2 && angles_cam[2]<= -angles_ref[2] + delta/2)){
        console.log('match1')
        if (cam_face.expression.toString() === faces_arr[selected].expression.toString()) {
            console.log('match2')
            swap_face(cam_face,faces_arr[selected])
            selected = -1
        }
    }
}
function equalize_hist(){
    let imgElement = document.getElementById("ImgViewImage"); // img element with ImgViewImage id
    let src = cv.imread(imgElement);
    let dst = new cv.Mat();
    let hsvPlanes = new cv.MatVector();
    let mergedPlanes = new cv.MatVector();
    cv.cvtColor(src, src, cv.COLOR_RGB2HSV, 0);
    cv.split(src, hsvPlanes);
    let H = hsvPlanes.get(0);
    let S = hsvPlanes.get(1);
    let V = hsvPlanes.get(2);
    cv.equalizeHist(V, V);
    mergedPlanes.push_back(H);
    mergedPlanes.push_back(S);
    mergedPlanes.push_back(V);
    cv.merge(mergedPlanes, src);
    cv.cvtColor(src, dst, cv.COLOR_HSV2RGB, 0);
    cv.imshow("canvasOutput", dst); // canavas element with canvasOutput id
    src.delete();
    dst.delete();
    hsvPlanes.delete();
    mergedPlanes.delete();
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
function swap_face(source_obj, target_obj) {
    // const source = source_obj.image
    const target = ref_img
    console.log(target.width)
    const source_points = source_obj.points
    const target_points = target_obj.points
    let tris1 = [], tris2 = [];
    for (let i in TRIANGULATION){
        // console.log(i+ ' '+ TRIANGULATION[i])
        let tri1 = [], tri2 = [];
        const tris_indices = TRIANGULATION[i]
        const tris_indices_mirror = TRIANGULATION_MIRROR[i]
        for (let j = 0; j < 3; j++) {
            tri1.push([source_points[tris_indices_mirror[j]][0],source_points[tris_indices_mirror[j]][1]])
            tri2.push([target_points[tris_indices[j]][0],target_points[tris_indices[j]][1]])
        }
        tris1.push(tri1)
        tris2.push(tri2)
    }

    // const deeep = deep()
    // deeep
    //     .then(value => {
    //     console.log(value);
    //     })
    //     .catch(function (error) {
    //         console.log("Something went wrong!");
    // });
    let target_mat = cv.imread(target)

    // cv.imshow(canvas, warp_triangle(cam_mat, target_mat, tris1[19], tris2[19]))
    for (let i in tris1) {
        let dst = warp_triangle(cam_mat, target_mat, tris1[i], tris2[i])

        target_mat = dst.clone()
        cv.imshow(canvas, target_mat)
        dst.delete()
    }
    target_mat.delete()
}
// deepai.setApiKey('1fdf0fa7-50b8-43a2-a2cb-56e9686120aa');
//
// async function deep() {
//     let resp = await deepai.callStandardApi("CNNMRF", {
//         content: video,
//         style: ref_img.src,
//     });
//     console.log('resp');
// }
function warp_triangle(img1, target, t1, t2 ){
    let bb1={
        minX: Math.min(t1[0][0], t1[1][0], t1[2][0]),
        minY: Math.min(t1[0][1], t1[1][1], t1[2][1]),
        maxX: Math.max(t1[0][0], t1[1][0], t1[2][0]),
        maxY: Math.max(t1[0][1], t1[1][1], t1[2][1]),
    }
    bb1.w= bb1.maxX - bb1.minX
    bb1.h= bb1.maxY - bb1.minY

    let bb2={
        minX: Math.min(t2[0][0], t2[1][0], t2[2][0]),
        minY: Math.min(t2[0][1], t2[1][1], t2[2][1]),
        maxX: Math.max(t2[0][0], t2[1][0], t2[2][0]),
        maxY: Math.max(t2[0][1], t2[1][1], t2[2][1]),
    }
    bb2.w= bb2.maxX - bb2.minX
    bb2.h= bb2.maxY - bb2.minY
    if ( bb1.w>> 0 || bb1.h >>0 || bb2.w>> 0 || bb2.h >>0){
        const r1 = new cv.Rect(bb1.minX,bb1.minY,bb1.w,bb1.h);
        const r2 = new cv.Rect(bb2.minX,bb2.minY,bb2.w,bb2.h);
        let roi_mask_t2_inv = new cv.Mat()

        // triangles masks from arrays
        // let mask_t1 = mask_from_array(t1, cam_face.w,cam_face.h)
        let mask_t2 = mask_from_array(t2, target.cols, target.rows)
        // cut triangles & masks Rois
        let roi_t1 = img1.roi(r1)
        let roi_t2 = target.roi(r2)
        // let roi_mask_t1 = mask_t1.roi(r1)
        let roi_mask_t2 = mask_t2.roi(r2)
        //convert to 1 channel gray
        // cv.cvtColor(roi_mask_t1,roi_mask_t1, cv.COLOR_RGBA2GRAY, 0)
        cv.cvtColor(roi_mask_t2,roi_mask_t2, cv.COLOR_RGBA2GRAY, 0)
        // inverse Roi mask t2
        cv.bitwise_not(roi_mask_t2,roi_mask_t2_inv)
        //
        // let masked_t1 = new cv.Mat()
        // cv.bitwise_and(roi_t1, roi_t1, masked_t1, roi_mask_t1)
        //normalize points , inverting x values in t1
        let n_t1 = [], n_t2 = []
        for (let i=0; i<3; i++){
            let new_point1 = [Math.round((1-(t1[i][0]- bb1.minX)/(bb1.maxX-bb1.minX)) * bb2.w), Math.round(((t1[i][1] - bb1.minY)/(bb1.maxY-bb1.minY)) * bb2.h)]
            n_t1.push(new_point1)
            let new_point2 = [Math.round(((t2[i][0]- bb2.minX)/(bb2.maxX-bb2.minX)) * bb2.w), Math.round(((t2[i][1]- bb2.minY)/(bb2.maxY-bb2.minY)) * bb2.h)]
            n_t2.push(new_point2)
        }
        // cv.flip(masked_t1, masked_t1, 1)
        let srcTri = cv.matFromArray(3, 1, cv.CV_32FC2, n_t1.flat());
        let dstTri = cv.matFromArray(3, 1, cv.CV_32FC2, n_t2.flat());
        let dsize = new cv.Size(bb2.w, bb2.h);
        let M = cv.getAffineTransform(srcTri, dstTri);
        M.data64F[2] += (M.data64F[0]+M.data64F[1]-1)/2
        M.data64F[5] += (M.data64F[3]+M.data64F[4]-1)/2
        let warped = roi_t2.clone()
        cv.resize(roi_t1, roi_t1, dsize, 0, 0, cv.INTER_AREA )
        cv.flip(roi_t1,roi_t1, 1)
        cv.warpAffine(roi_t1, warped, M, dsize, cv.INTER_AREA, cv.BORDER_REFLECT_101 , new cv.Scalar());

        let mix = mask_img( roi_t2, warped, roi_mask_t2,roi_mask_t2_inv)
        let dst = target.clone();
        for (let i = 0; i < roi_t2.rows; i++) {
            for (let j = 0; j < roi_t2.cols; j++) {
                dst.ucharPtr(i+bb2.minY, j+bb2.minX)[0] = mix.ucharPtr(i, j)[0];
                dst.ucharPtr(i+bb2.minY, j+bb2.minX)[1] = mix.ucharPtr(i, j)[1];
                dst.ucharPtr(i+bb2.minY, j+bb2.minX)[2] = mix.ucharPtr(i, j)[2]
            }
        }
        // target = dst
        // roi_mask_t1.delete(); mask_t1.delete();
        // cv.imshow(canvas, target)
        mask_t2.delete(); roi_t1.delete();roi_t2.delete();
        roi_mask_t2.delete(); roi_mask_t2_inv.delete(); mix.delete()
        srcTri.delete(); dstTri.delete();  warped.delete();M.delete();// masked_t1.delete();
        return dst
    }

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
async  function detectFaces(){

    cam_face = new Face('cam', video)
    cam_face.w=video.width;
    cam_face.h=video.height;

    const cam_promise = calc_lmrks(video, 'cam')
    cam_promise
        .then((value) => {
            // console.log('value');
            // console.log(value);
            cam_face.lmrks = value[0];//This is a fulfilled promise  👈
            cam_face.points = value[1];
            cam_face.bb = value[2];
            cam_face.angles = value[3];
            cam_face.hull = convex_hull(value[1], value[2]);
            cam_face.n_points = cam_face.normalize_array(value[1])
            cam_face.expression = check_expression(value[1])
            // console.log(bb)
            if (selected>=0) {
                draw_mask_on_ref(video, cam_face)
                match(cam_face.angles, faces_arr[selected].angles)
            }
            // cv_draw(video, video_bb)

            // console.log(angles)
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
        draw_mask_on_ref(video, cam_face)
        // drawHull(ref_img, selected);
    }
}
init();
accessCamera();
//
// const cv = document.cv
// this event will be  executed when the video is loaded
// window.addEventListener('DOMContentLoaded', (event) => {
//     console.log('DOM fully loaded and parsed ' + ref_img.src);
//     // cv.rectangle(ref_img.src, (0,0), (30,30), (255,0,0),5 )
//
// });
window.addEventListener('resize', resizeCanvas, false);
video.addEventListener('loadeddata',function() {
    body.classList.add('loaded')
    setInterval(detectFaces, 100);})
// video.addEventListener("loadeddata", async () => {
//     setInterval(detectFaces, 100);
// });

////////////////////////////////////////

