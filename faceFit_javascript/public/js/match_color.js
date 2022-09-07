// //*** IMPLEMENTATION OF A FURTHER ENHANCED ADAPTATION
// //*** OF THE REINHARD COLOUR TRANSFER METHOD.
// //    See 'main' routine for further details.
// //
// // Copyright © Terry Johnson, January 2020
// // Revised 27/05/2021 (Version 5)
// // https://github.com/TJCoding
//
// #include <opencv2/highgui/highgui.hpp>
// #include <opencv2/photo/photo.hpp>
//
// // Declare functions
// cv::Mat CoreProcessing(cv::Mat targetf, cv::Mat sourcef,
//     float CrossCovarianceLimit,
//     int   ReshapingIterations,
//     float ShaderVal);
// cv::Mat adjust_covariance(cv::Mat Lab[3], cv::Mat sLab[3],
//     float covLim);
// cv::Mat ChannelCondition(cv::Mat tChan, cv::Mat sChan);
// cv::Mat SaturationProcessing(cv::Mat targetf, cv::Mat savedtf,
//     float SatVal);
// cv::Mat FullShading(cv::Mat targetf, cv::Mat savedtf, cv::Mat sourcef,
//     bool ExtraShading, float ShadeVal);
// cv::Mat FinalAdjustment(cv::Mat targetf, cv::Mat savedtf,
//     float TintVal, float ModifiedVal);
// cv::Mat convertTolab  (cv::Mat input);
// cv::Mat convertFromlab(cv::Mat input);
// let target = cv.imread('/images/image04.jpg')
// let source = cv.imread('/images/image11.jpg')


function match_c(target, source) {
//  Transfers the colour distribution from the source image to the target image by matching the mean and standard
//  deviation and the colour cross correlation in the L-alpha-beta colour space.
//  Additional image refinement options are also provided.

//	The implementation is an enhancement of a method described in "Color Transfer between Images" paper by Reinhard et al., 2001,
//  but with additional processing options as follows.

// PROCESSING OPTIONS.
//  OPTION 1
//  There is an option to match the cross correlation between the colour channels 'alpha' and 'beta'.  Full matching, no
//  matching or restricted matching may be specified. (See the note at the end of the code).

//  OPTION 2
//  There is an option to further match the distribution of the target 'alpha' and 'beta' colour channels to the corresponding
//  source channels by iteratively adjusting the higher order characteristics of the channel data (skewness and kurtosis)
//  in addition to matching the mean and standard deviation values. The number of iterative adjustments can be specified for this
//  'reshaping' option. A zero value specifies no reshaping processing and a non-zero value specifies the total number of reshaping
//  iterations to be implemented.

//  OPTION 3
//  There is an option to pare back excess colour saturation resulting from the colour transfer processing.

//  OPTION 4
//  There is an option to retain the shading of the target image so that the processing is implemented as a pure colour transfer
//  or to adopt the shading of the source image or to select an intermediate shading.

//  OPTION 5
//  There is an option to further adjust the shading of the output image to match the target image, the source image or an
//  intermediate image (as selected under Option 4) by directly matching distributions in the corresponding grey shade images.

//  OPTION 6
//  There is an option to modify the colour tint of the final image.

//  OPTION 7
//  There is an option to mix the final image with the initial image, so that only part modification occurs.

// ##############################################################################################################
// ########################################  PROCESSING SELECTIONS  #############################################
// ##############################################################################################################
    // Select the processing options in accordance with the preceding descriptions.

    let CrossCovarianceLimit     = 0.5;    // Option 1 (Default is '0.5')
    let ReshapingIterations      = 1;      // Option 2 (Default is '1')
    let PercentSaturationShift   = -1.0;   // Option 3 (Default is -1.0)
    let PercentShadingShift      = 50.0;   // Option 4 (Default is 50.0)
    let ExtraShading             = true;   // Option 5 (Default is 'true')
    let PercentTint              = 100.0;  // Option 6 (Default is 100.0)
    let PercentModified          = 100.0;  // Option 7 (Default is 100.0)

    //  Setting CrossCovarianceLimit to 0.0 inhibits cross covariance processing.
    //  Setting ReshapingIterations to 0, inhibits reshaping processing.
    //  Setting PercentShadingShift to 0, retains the target image saturation.
    //  Setting PercentShadingShift to 0, retains the target image shading.
    //  Setting ExtraShading to 'false' reverts to simple shading.
    //  Setting PercentTint to '0', gives a monochrome image.
    //  Setting PercentModified to '0', retains the target image in full.

    //  For each of the percentage parameters, defined above, a setting of '100' allows the full processing effect.
    //  A setting of '0' suppresses the particular processing effect. Intermediate values give an intermediate outcome
    //  Percentages below '0' or above '100' are permitted although the outcome may not always be pleasing.
    //  (See note below for saturation shift.)
    //
    // Note:
    // If a negative percentage is set for saturation shift, then the actual percentage is determined automatically from
    // the image properties.


    // Specify the image files that are to be processed, where 'source image' provides the colour scheme that is to be
    // applied to 'target image'.

    // let targetname = "./images/image04.jpg";
    // let sourcename = "./images/image11.jpg";

// ###############################################################################################################
// ###############################################################################################################
// ###############################################################################################################

    // Read in the images and convert to floating point, saving a copy of the target image for later.
    let target_mat = cv.imread(target);
    // let source = cv.imread(sourcename);
    // console.log(typeof source + ' ' +typeof target)
    let targetf = new cv.Mat(target_mat.rows, target_mat.cols,cv.CV_32FC3);
    let sourcef = new cv.Mat(source.rows, source.cols,cv.CV_32FC3);
    target_mat.convertTo(targetf, cv.CV_32FC3, 1.0);// /255
    source.convertTo(sourcef, cv.CV_32FC3, 1.0);// /255
    let savedtf = targetf.clone();
    console.log('1 image width: ' + targetf.cols + '\n' +
        'image height: ' + targetf.rows + '\n' +
        'image size: ' + targetf.size().width + '*' + targetf.size().height + '\n' +
        'image depth: ' + targetf.depth() + '\n' +
        'image channels ' + targetf.channels() + '\n' +
        'image type: ' + target_mat.type() + '\n' +
        'image type: ' + targetf.type() + '\n');
    // Implement augmented "Reinhard Processing" in L-alpha-beta colour space.
    targetf = CoreProcessing(targetf, sourcef, CrossCovarianceLimit, ReshapingIterations, PercentShadingShift/100.0);
    cv.cvtColor(sourcef,sourcef,cv.COLOR_BGR2GRAY, 0); // Only need mono hereafter.

    // Implement image refinements where a change is specified.
    SaturationProcessing(targetf, savedtf, PercentSaturationShift/100.0);
    targetf = FullShading(targetf, savedtf, sourcef, ExtraShading, PercentShadingShift/100.0);
    targetf = FinalAdjustment(targetf,savedtf, PercentTint/100.0, PercentModified/100.0);

    //  Convert the processed image to integer format.
    let result = new cv.Mat();

    targetf.convertTo(result, cv.CV_8UC3, 255);
    // Display and save the final image.
    cv.imshow("processed image",result);
    cv.imwrite("images/processed.jpg", result);

    // Display image until a key is pressed.
    cv.waitKey(0);
    return 0;
}

function CoreProcessing(targetf, sourcef, CrossCovarianceLimit, ReshapingIterations, ShaderVal){
    // Implements augmented "Reinhard Processing" in L-alpha-beta colour space.
    // First convert the images from the BGR colour space to the L-alpha-beta colour space.
    // Estimate the mean and standard deviation of colour channels. Split the target and source
    // images into colour channels and standardise the distribution within each channel.
    // The standardised data has zero mean and unit standard deviation.
    
    let Lab = new cv.MatVector();
    let sLab = new cv.MatVector();
    // let tmean, tdev, smean, sdev;
    console.log('image width: ' + targetf.cols + '\n' +
        'image height: ' + targetf.rows + '\n' +
        'image size: ' + targetf.size().width + '*' + targetf.size().height + '\n' +
        'image depth: ' + targetf.depth() + '\n' +
        'image channels ' + targetf.channels() + '\n' +
        'image type: ' + targetf.type() + '\n');
    targetf = convertTolab(targetf);
    sourcef = convertTolab(sourcef);
    console.log(targetf)
    //cv::meanStdDev(targetf, tmean, tdev);
    //cv::meanStdDev(sourcef, smean, sdev);
    let tf_mean = new cv.Mat(1,4,cv.CV_64F);
    let tf_dev = new cv.Mat(1,4,cv.CV_64F);
    cv.meanStdDev(targetf, tf_mean, tf_dev);
    let sf_mean = new cv.Mat(1,4,cv.CV_64F);
    let sf_dev = new cv.Mat(1,4,cv.CV_64F);
    cv.meanStdDev(sourcef, sf_mean, sf_dev);
    console.log(tf_mean.doubleAt(0,0))
    // cv.imwrite(output2, targetf)
    console.log('image width: ' + targetf.cols + '\n' +
        'image height: ' + targetf.rows + '\n' +
        'image size: ' + targetf.size().width + '*' + targetf.size().height + '\n' +
        'image depth: ' + targetf.depth() + '\n' +
        'image channels ' + targetf.channels() + '\n' +
        'image type: ' + targetf.type() + '\n');
    cv.split(targetf, Lab);
    // cv.split(sourcef,sLab);
    // for(let el in targetf){
    //     console.log (targetf[5][0])
    // }
    // const t_length = targetf.length
    // const s_length = sourcef.length
    // let L=[], a=[], b=[];
    // for (let i=0; i < t_length/3; i++){
    //     let id1 = i * 3;
    //     let id2 = id1 + 1;
    //     let id3 = id1 + 2;
    //     L.push(targetf[id1])
    //     console.log(L)
    //     a.push(targetf[id2])
    //     b.push(targetf[id3])
    // }
    // let sL=[], sa=[], sb=[];
    // for (let i=0; i < s_length/3; i++){
    //     let id1 = i*3;
    //     let id2 = id1 + 1;
    //     let id3 = id1 + 2;
    //     sL.push(sourcef[id1])
    //     sa.push(sourcef[id2])
    //     sb.push(sourcef[id3])
    // }
    // Lab[0] = L
    // Lab[1] = a
    // Lab[2] = b

    Lab[0] = (Lab[0]-tf_mean[0])/tf_dev[0];
    Lab[1] = (Lab[1]-tf_mean[1])/tf_dev[1];
    Lab[2] = (Lab[2]-tf_mean[2])/tf_dev[2];

    // sLab[0]=sL
    // sLab[1]=sa
    // sLab[2]=sb
    sLab[0] = (sLab[0]-sf_mean[0])/sf_dev[0];
    sLab[1] = (sLab[1]-sf_mean[1])/sf_dev[1];
    sLab[2] = (sLab[2]-sf_mean[2])/sf_dev[2];
    console.log(Lab[0])

    // Implement first phase of reshaping for the colour channels when one or more iteration is specified.
    let jcount = ReshapingIterations;
    while (jcount > Math.ceil((ReshapingIterations+1)/2)) {
        Lab[1] = ChannelCondition(Lab[1],sLab[1]);
        Lab[2] = ChannelCondition(Lab[2],sLab[2]);
        jcount--;
    }
    // Implement cross covariance processing. (null if CrossCovarianceLimit=0.0)
    targetf = adjust_covariance(Lab, sLab, CrossCovarianceLimit );
    cv.split(targetf,Lab);

    // Implement second phase of reshaping
    while (jcount>0)
    {
        Lab[1]=ChannelCondition(Lab[1],sLab[1]);
        Lab[2]=ChannelCondition(Lab[2],sLab[2]);
        jcount--;
    }

    // Rescale the previously standardised colour channels so that the means and standard deviations now match
    // those of the source image.
    Lab[1] = Lab[1] * sf_dev[1] + sf_mean[1];
    Lab[2] = Lab[2] * sf_dev[2] + sf_mean[2];

    // Rescale the lightness channel (channel 0) in accordance with the specified percentage shading shift.
    Lab[0] = Lab[0] * (ShaderVal*sf_dev[0]+(1.0-ShaderVal)*tf_dev[0]) + ShaderVal * sf_mean[0] + (1.0 - ShaderVal) * tf_mean[0];

    // Merge channels and convert back to BGR colour space.
    let resultant = new cv.Mat();
    cv.merge(Lab, resultant);
    resultant = convertFromlab(resultant);
    return resultant;
}

function adjust_covariance(Lab, sLab, covLim) {
    // This routine adjusts colour channels 2 and 3 of the image within the L-alpha-beta colour space.

    // The channels each have zero mean and unit standard deviation but their cross correlation value will not normally
    // be zero.
    //
    // The processing reduces the cross correlation between the channels to zero but then reintroduces correlation such
    // that the new cross correlation value matches that for the source image.
    //
    // Throughout these manipulations the mean channel values are maintained at zero and the standard deviations are
    // maintained as unity.
    //
    // The manipulations are based upon the following relationship.
    //
    // Let z1 and z2 be two independent (zero correlation) variables with zero means and unit standard deviations.
    // It can be shown that variables a1 and a2 have zero means, unit standard deviations, and mutual cross correlation
    // 'R' when:
    // a1=sqrt((1+R)/2)*z1 + sqrt((1-R)/2)*z2
    // a2=sqrt((1+R)/2)*z1 - sqrt((1-R)/2)*z2
    //
    // The above relationships are applied inversely to derive uncorrelated standardised colour channels variables from
    // the standardised but correlated input channels.
    //
    // The above relationships are then applied directly to obtain standardised correlated colour channels with correlation
    // matched to that of the source image colour channels.
    //
    // Original processing method attributable to Dr T E Johnson Sept 2019.

    // Declare variables
    let tcrosscorr, scrosscorr, W1, W2, norm, smean, sdev, temp2;
    let temp1 = new cv.Mat();

    // No processing required if 'covLim' set to zero.
    if(covLim!=0.0) {
        // Compute the correlation for the target image colour channels.
        // The correlation between the standardised variables (zero mean, unit standard deviation) can be computed as
        // the mean cross product for the two channels.
        temp1 = multiply_arrays(Lab[1],Lab[2]);
        // temp2=cv::mean(temp1);
        console.log(temp1)
        temp2 = temp1.flat()
        console.log(temp2)
        temp2 = temp2.reduce((a, b) => a + b, 0) / temp2.length;
        tcrosscorr =temp2[0]
        console.log(temp2 +' '+ temp2[0])
        // Compute the correlation for the source image colour channels.
        // cv::multiply(sLab[1],sLab[2],temp1);
        temp1 = multiply_arrays(sLab[1],sLab[2]);
        temp2= temp2.reduce((a, b) => a + b, 0) / temp2.length; // MEAN
        scrosscorr =temp2[0];

        // Adjust the correlation between the
        // standardised input channel values.
        W1= 0.5 * Math.sqrt((1+scrosscorr)/(1+tcrosscorr)) + 0.5 * Math.sqrt((1-scrosscorr)/(1-tcrosscorr));
        W2= 0.5 * Math.sqrt((1+scrosscorr)/(1+tcrosscorr)) - 0.5 * Math.sqrt((1-scrosscorr)/(1-tcrosscorr));

        // Limit the size of W2 if required.
        // This limits the proportional amount by which a given colour channel can be augmented by the energy from the
        // other colour channel.
        if (Math.abs()(W2) > covLim * Math.abs()(W1))
        {
            W2 = copysign(covLim * W1, W2);
            norm = 1.0 / Math.sqrt(W1 * W1 + W2 * W2 + 2 * W1 * W2 * tcrosscorr);
            W1 = W1 * norm;
            W2 = W2 * norm;
        }
        let z1 = Lab[1].clone();

        Lab[1] = W1 * z1 + W2 * Lab[2];
        Lab[2] = W1 * Lab[2] + W2*z1;
    }
    cv.merge(Lab,temp1);
    return temp1;
}

function multiply_arrays(arr1, arr2) {
    let result = []
    for (let el in arr1){
        let prod= arr1[el]*arr2[el]
        result.push(prod)
    }
    return result
}

function divide_arrays(arr1, arr2){
    let result = []
    for (let el in arr1){
        let div= arr1[el]/arr2[el]
        result.push(div)
    }
    return result
}

function exp(src){
    let dst = []
    for (let el in src){
        let new_el = Math.exp(src[el])
        dst.push(new_el)
    }
    return dst
}

function ChannelCondition(Chan, sChan) {
// Modifies the distribution of values in 'Chan' to more closely match the distribution of those in 'sChan'.
// Separate matching operations are performed for values above and below the mean.  The input channels have
// been standardised so the mean is equal to zero.
// Original processing method attributable to Dr T E Johnson Oct 2020.

    // Declare variables
    // Computations use weighted data values.
    // 'wval' is the tuning constant for the weighting function.
    let mask = new cv.Mat()
    let ChanU = new cv.Mat()
    let ChanL = new cv.Mat();
    let WU = new cv.Mat()
    let WL = new cv.Mat()
    let smeanU, smeanL, wmean, tmeanU, tmeanL, tmean, tdev;
    let k
    let wval = 0.25;

    // sChan is processed before Chan because Chan data is used in later processing.

    // Processing for upper 'sChan'.

    // Determine the mask for selecting data values above zero.
    cv.threshold(sChan, mask, 0, 1, cv.THRESH_BINARY);
    mask.convertTo(mask, cv.CV_8UC1);

    // Compute the weighting function for values above zero.
    // (Zero is the mean value of the input channel).
    // The weighting function is zero for sChan values equal to zero and unity for large values of sChan.
    // let mask = cv.Mat.zeros(image.cols, image.rows, cv.CV_8UC1);
    // let mean = new cv.Mat;
    // let std = new cv.Mat;
    // cv.meanStdDev(image, mean, std, mask);
    // smeanU = mean(sChan,mask);
    smeanU = meanAndStd(sChan,mask)[0]
    WU = exp(-sChan * wval / smeanU);
    WU = (1 - WU).mul(1 - WU);
    wmean = meanAndStd(WU,mask)[0]
    // Compute deviation from the mean and raise to the power 4 so as to address kurtosis.
    ChanU = arrayPower(sChan,4)
    // Find the weighted average of the fourth power of the deviations.
    smeanU =meanAndStd(WU.mul(ChanU),mask)[0]/wmean;

    // Processing for lower 'sChan'.

    // As for upper processing but values are selected by applying the complementary masking function (1-mask).
    smeanL = meanAndStd(sChan,(1-mask))[0];
    WL = exp(-sChan * wval / smeanL);
    WL = (1-WL).mul(1-WL);
    wmean=meanAndStd(WL,1-mask);
    ChanL = arrayPower(sChan,4);
    smeanL=meanAndStd(WL.mul(ChanL),1-mask)[0] / wmean;

    // Processing for upper 'Chan'
    cv.threshold(Chan, mask, 0, 1, cv.THRESH_BINARY);
    mask.convertTo(mask, cv.CV_8UC1);

    tmeanU = meanAndStd(Chan, mask)[0];
    WU = exp(-Chan * wval / tmeanU);
    WU = (1-WU).mul(1-WU);
    wmean = meanAndStd(WU, mask)[0];
    ChanU = arrayPower(Chan,4);
    tmeanU = meanAndStd(WU.mul(ChanU), mask)[0] / wmean;

    // Processing for lower 'Chan'

    tmeanL=meanAndStd(Chan,(1-mask))[0];
    WL = exp(-Chan*wval/tmeanL);
    WL = (1 - WL).mul(1 - WL);
    wmean = meanAndStd(WL,1-mask)[0];
    ChanL = arrayPower(Chan,4);
    tmeanL = meanAndStd(WL.mul(ChanL),1-mask)[0]/wmean;

    // Modify the upper 'Chan' values

    // Compute the ratio of the weighted fourth power for 'sChan' relative to that for 'Chan' and then take the fourth root.
    // The resultant is used to apply a shift to the 'Chan' data where the shift is a function of the data deviation.
    // No shift is applied to small values and full shift to large values.
    k = Math.sqrt(Math.sqrt(smeanU/tmeanU));
    ChanU = (1 + WU * (k - 1)).mul(Chan);

    // Similarly modify the lower 'Chan' values.
    k = Math.sqrt(Math.sqrt(smeanL/tmeanL));
    ChanL = (1 + WL * (k- 1)).mul(Chan);

    // Combine the upper and lower 'Chan'values to form a whole
    Chan = cv.Mat.zeros(Chan.rows, Chan.cols, cv.CV_32FC1);
    cv.add(Chan, ChanU, Chan, mask);//add dtype
    cv.add(Chan, ChanL, Chan, 1-mask);

    // Re-standardise the modified 'Chan' data before it is fed back.
    cv.meanStdDev(Chan, tmean, tdev);
    Chan = (Chan - tmean[0]) / tdev[0]; //??????????????????????????
    cv.meanStdDev(Chan, tmean, tdev);

    return Chan;
}

function arrayPower(nums, K) {

    // Loop is used for perforimg power of K
    for (let i = 0; i < nums.length; i++) {
        nums[i] = Math.pow(nums[i], K);
    }
    return nums
}

function meanAndStd(arr, mask){
    let mean = new cv.Mat;
    let std = new cv.Mat;
    cv.meanStdDev(arr, mean, std, mask);
    return [mean, std]
}
//
function SaturationProcessing(targetf, savedtf, SatVal) {
    // This routine allows a reduction of colour saturation to an extent specified by the parameter 'SatVal'.
    // An image that is subject to full colour transfer can often exhibit excessive colour saturation. This function
    // allows a scaling back of saturation.

    // The idea of saturation processing is to adjust the saturation characteristics of the processed image to match the
    // saturation characteristics of an artificially constructed image whose saturation characteristics are considered
    // desirable.

    // Implement a saturation change unless 100% saturation is specified.
    if (SatVal!=1)
    {
        let temp = new cv.Mat()
        let mask = new cv.Mat()
        let Hsv = new cv.Mat()
        let tmpHsv = new cv.Mat()
        let tmean= new cv.Mat()
        let tdev= new cv.Mat()
        let tmpmean= new cv.Mat()
        let tmpdev= new cv.Mat()

        // Colour saturation will be computed in accordance with the definition used for the HSV colour space.
        cv.cvtColor(targetf,targetf,cv.COLOR_BGR2HSV);
        cv.cvtColor(savedtf,temp,cv.COLOR_BGR2HSV);
        cv.split(targetf,Hsv);
        cv.split(temp,tmpHsv);

        if(SatVal<0)
        {
            //  'SatVal' is less than 0, then compute 'SatVal' as the ratio of the largest saturation value in the
            //  original image to the largest value in processed image.
            let minmax1, minmax2;
            minmax1 = minMaxIdx(Hsv);
            minmax2 = minMaxIdx(tmpHsv);
            SatVal=minmax2[1]/minmax1[1];
        }

        // Compute a weighted mix of the processed target saturation channel  and the original image saturation channel
        // to define an initial reference saturation channel.
        cv.addWeighted(Hsv, SatVal,tmpHsv, 1-SatVal, 0.0, tmpHsv);

        // The initial reference saturation channel values will apply only to those pixels where the saturation in the
        // processed image exceeds that in the original target image.
        cv.threshold((Hsv - tmpHsv),mask,0,1,cv.THRESH_BINARY);
        mask.convertTo(mask,cv.CV_8UC1);

        // Create a new reference saturation channel which is taken from the initial reference saturation channel except
        // where the mask function indicates that the saturation value of the original image should be used.
        // This gives a modified reference saturation channel.
        temp = cv.Mat.zeros(targetf.rows, targetf.cols, cv.CV_32FC1);
        cv.add(temp,tmpHsv,temp,mask);
        cv.add(temp,Hsv,tmpHsv,(1-mask));

        // Now match the mean and standard deviation of the saturation channel for the processed image channel to the
        // mean and standard deviation of the modified reference saturation channel. This give the final saturation
        // channel which is the output from the saturation processing
        cv.meanStdDev(Hsv, tmean, tdev);
        cv.meanStdDev(tmpHsv, tmpmean, tmpdev);
        Hsv[1] = (Hsv - tmean) / tdev;
        Hsv[1] = Hsv * tmpdev + tmpmean;
        cv.merge(Hsv, targetf);
        cv.cvtColor(targetf,targetf,cv.COLOR_HSV2BGR);
    }
    return targetf;
}

function minMaxIdx(arr){
    const max = Math.max(...arr);
    const min = Math.min(...arr);
    const minId = arr.indexOf(min)
    const maxId = arr.indexOf(max)
    return [minId, maxId]
}

function maxArray(arr1, arr2){
    let dst = cv.Mat.Zeros(arr1.rows,arr1.cols, arr1.type)
    for (let el in arr1){
        dst[el] = Math.max([arr1[el], arr2[el]])
    }
    return dst
}

function FullShading(targetf, savedtf, sourcef, ExtraShading, ShaderVal) {
    // Matches the grey shade distribution of the modified target image to that of a notional shader image which is a
    // linear combination of the original target and source image as determined by the value of 'ShaderVal'.

    if (ExtraShading)
    {
        let greyt = new cv.Mat()
        let greys = new cv.Mat()
        let greyp = new cv.Mat()
        let chans = new cv.Mat()
        let smean = new cv.Mat()
        let tmean = new cv.Mat()
        let sdev  = new cv.Mat()
        let tdev  = new cv.Mat()

        // Compute the grey shade images for the target, processed and source images.
        cv.cvtColor(savedtf, greyt, cv.COLOR_BGR2GRAY);
        cv.cvtColor(targetf ,greyp, cv.COLOR_BGR2GRAY);
        sourcef.copyTo(greys);// Already converted.
        greys = sourcef.clone()
        // Standardise the greyshade images for the source and target.
        cv.meanStdDev(greys, smean, sdev);
        cv.meanStdDev(greyt, tmean, tdev);
        greyt = (greyt - tmean)/tdev;

        // Rescale the previously standardised grey shade target image so that the means and standard deviations now
        // match those of the notional shader image.
        greyt = greyt * (ShaderVal * sdev + (1.0 - ShaderVal) * tdev) +ShaderVal * smean + (1.0 - ShaderVal) * tmean;

        // Rescale each of the colour channels of the processed image identically so that in grey shade the processed
        // image more closely matches the grey shading of the nominated goal image.
        let min_mat = new cv.Mat(greyp.size(), cv.CV_32FC1,1/255.0);
        greyp = maxArray(greyp, min_mat); // Guard against zero divide;
        // Guard against negative values;
        greyt = maxArray(greyt, cv.Mat(greyt.size(), cv.CV_32FC1, 0.0));



        cv.split(targetf,chans);
        chans[0] = divide_arrays(chans[0],greyp);
        chans[0] = chans[0].mul(greyt);
        chans[1] = divide_arrays(chans[1],greyp);
        chans[1] = chans[1].mul(greyt);
        chans[2] = divide_arrays(chans[2],greyp);
        chans[2] = chans[2].mul(greyt);
        cv.merge(chans, targetf);
    }

    return targetf;
}

function FinalAdjustment(targetf, savedtf, TintVal, ModifiedVal) {
// Implements a change to the tint of the final image and to its degree of modification if a change is specified.

    // If 100% tint not specified then compute a weighted average of the processed image and its grey scale representation.
    if (TintVal !== 1.0) {
        let grey = new cv.Mat();
        let BGR = new cv.Mat(); ////verifica tipo di mat

        cv.cvtColor(targetf, grey, cv.COLOR_BGR2GRAY);
        cv.split(targetf, BGR);
        BGR[0] = TintVal * BGR[0] + (1.0 - TintVal) * grey;
        BGR[1] = TintVal * BGR[1] + (1.0 - TintVal) * grey;
        BGR[2] = TintVal * BGR[2] + (1.0 - TintVal) * grey;
        cv.merge(BGR,targetf);
    }

    // If 100% image modification not specified then compute a weighted average of the processed image and the original
    // target image.
    if (ModifiedVal !== 1.0) {
        targetf = ModifiedVal * targetf + (1.0 - ModifiedVal) * savedtf;
    }

    return targetf;
}

//
//
// // ##########################################################################
// // ##### IMPLEMENTATION OF L-ALPHA-BETA FORWARD AND INVERSE TRANSFORMS ######
// // ##########################################################################
// // Coding taken from https://github.com/ZZPot/Color-transfer
// // Credit to 'ZZPot'.
// // I take responsibility for any issues arising my adaptation.
//
//
// // Define the transformation matrices for L-alpha-beta transformation.
// cv::Mat RGB_to_LMS = (cv::Mat_<float>(3,3) <<	0.3811f, 0.5783f, 0.0402f,
//     0.1967f, 0.7244f, 0.0782f,
//     0.0241f, 0.1288f, 0.8444f);
// float i3 = 1/sqrt(3), i6 = 1/sqrt(6), i2 = 1/sqrt(2);
// cv::Mat LMS_to_lab = (cv::Mat_<float>(3,3) <<	i3, i3, i3,
//     i6, i6, -2*i6,
//     i2, -i2, 0);
//
//
//

function convertTolab(input) {
    let img_RGBf = new cv.Mat(input.size(),cv.CV_32FC3);
    // let img_lms = new cv.Mat(input.size(),cv.CV_32FC3);
    let img_lab = new cv.Mat(input.size(),cv.CV_32FC3);

    // Swap channel order (so that transformation matrices can be used in their familiar form).
    // Then convert to float.
    //cv::cvtColor(input, img_RGB, CV_BGR2RGB);
    //img_RGB.convertTo(img_RGBf, CV_32FC3, 1.0/255.f);
    if (input.type()==29){
        cv.cvtColor(input, img_RGBf, cv.COLOR_BGRA2RGB)
    }
    else {
        cv.cvtColor(input, img_RGBf, cv.COLOR_BGR2RGB);
    }
    cv.cvtColor(img_RGBf, img_lab, cv.COLOR_RGB2Lab);
    // // Apply stage 1 transform.
    // cv::transform(img_RGBf, img_lms, RGB_to_LMS);
    //
    // // Define smallest permitted value and implement it.
    // const epsilon =1.0/255;
    // let min_scalar = (epsilon, epsilon, epsilon);
    // let min_mat = new cv.Mat(input.size(), CV_32FC3, min_scalar);
    // cv::max(img_lms, min_mat, img_lms); // just before log operation.
    //
    // // Compute log10(x)  as ln(x)/ln(10)
    // cv::log(img_lms,img_lms);
    // img_lms=img_lms/log(10);
    //
    // // Apply stage 2 transform.
    // cv::transform(img_lms, img_lab, LMS_to_lab);
    // cv.imshow(output2, img_RGBf)
    img_RGBf.delete();
    // console.log(img_lab)

    return img_lab;
}

function convertFromlab(input) {
    let img_lms = new cv.Mat(input.size(),  cv.CV_32FC3);
    let img_RGBf = new cv.Mat(input.size(),  cv.CV_32FC3);
    let img_BGRf = new cv.Mat(input.size(),  cv.CV_32FC3);
    let temp = new cv.Mat(LMS_to_lab.size(),cv.CV_32FC1);

    cv.cvtColor(input, img_RGBf, cv.COLOR_Lab2RGB);
    cv.cvtColor(img_RGBf, img_BGRf, cv.COLOR_RGB2BGR);
    // Apply inverse of stage 2 transformation.
    // cv::invert(LMS_to_lab,temp);
    // cv::transform(input, img_lms, temp);
    //
    // // Compute 10^x as (e^x)^(ln10)
    // cv::exp(img_lms,img_lms);
    // cv::pow(img_lms,(double)log(10.0),img_lms);
    //
    // // Apply inverse of stage 1 transformation.
    // cv::invert(RGB_to_LMS,temp);
    // cv::transform(img_lms, img_RGBf, temp);
    //
    // //  Revert channel ordering to BGR.
    // cv::cvtColor(img_RGBf, img_BGRf, cv.CV_RGB2BGR);

    return img_BGRf;
}

function lab2rgb(lab){
    let y = (lab[0] + 16) / 116,
        x = lab[1] / 500 + y,
        z = y - lab[2] / 200,
        r, g, b;

    x = 0.95047 * ((x * x * x > 0.008856) ? x * x * x : (x - 16/116) / 7.787);
    y = 1.00000 * ((y * y * y > 0.008856) ? y * y * y : (y - 16/116) / 7.787);
    z = 1.08883 * ((z * z * z > 0.008856) ? z * z * z : (z - 16/116) / 7.787);

    r = x *  3.2406 + y * -1.5372 + z * -0.4986;
    g = x * -0.9689 + y *  1.8758 + z *  0.0415;
    b = x *  0.0557 + y * -0.2040 + z *  1.0570;

    r = (r > 0.0031308) ? (1.055 * Math.pow(r, 1/2.4) - 0.055) : 12.92 * r;
    g = (g > 0.0031308) ? (1.055 * Math.pow(g, 1/2.4) - 0.055) : 12.92 * g;
    b = (b > 0.0031308) ? (1.055 * Math.pow(b, 1/2.4) - 0.055) : 12.92 * b;

    return [Math.max(0, Math.min(1, r)) * 255,
        Math.max(0, Math.min(1, g)) * 255,
        Math.max(0, Math.min(1, b)) * 255]
}

function rgb2lab(rgb){
    let r = rgb[0] / 255,
        g = rgb[1] / 255,
        b = rgb[2] / 255,
        x, y, z;

    r = (r > 0.04045) ? Math.pow((r + 0.055) / 1.055, 2.4) : r / 12.92;
    g = (g > 0.04045) ? Math.pow((g + 0.055) / 1.055, 2.4) : g / 12.92;
    b = (b > 0.04045) ? Math.pow((b + 0.055) / 1.055, 2.4) : b / 12.92;

    x = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047;
    y = (r * 0.2126 + g * 0.7152 + b * 0.0722) / 1.00000;
    z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883;

    x = (x > 0.008856) ? Math.pow(x, 1/3) : (7.787 * x) + 16/116;
    y = (y > 0.008856) ? Math.pow(y, 1/3) : (7.787 * y) + 16/116;
    z = (z > 0.008856) ? Math.pow(z, 1/3) : (7.787 * z) + 16/116;

    return [(116 * y) - 16, 500 * (x - y), 200 * (y - z)]
}


// // ##########################################################################
// // ##########################################################################
// // ##########################################################################
//
//
// // Notes on Cross Correlation Matching.
// // ====================================
// // Cross correlation matching is performed by operations of the
// // form.
// // Channel_alpha = W1*Channel_alpha + W2*Channel_beta
// // Channel_beta  = W1*Channel_beta  + W2*Channel_alpha
// // as determined by the value of CrossCovarianceLimit.
// //
// // If CrossCovarianceLimit = 0, W2=0 and no cross correlation
// // matching is performed.
// // If CrossCovarianceLimit > 0, W2 is clipped if necessary so
// // that it cannot lie outside the range
// // -CrossCovarianceLimit*W1 to +CrossCovarianceLimit*W1.
// // This mechanism may be used to guard against an overly large
// // correction term.
// //
// // Typically CrossCovarianceLimit might be set to 0.5
// //(for a maximum modification corresponding to 50%).







