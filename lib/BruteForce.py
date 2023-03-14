####
#### Brute-Force with OpenCV2
####

def BrForce(des1, des2, check, matching_distance, crossCheck_bool, matching_strategy, print_debug = True, ratio_thresh=0.8):
    if check == 'without_Lowe_ratio_test' and matching_distance=='L2':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck_bool)
        print(des1)
        print(des2)
        print(type(des1))
        print(type(des2))
        print(des1.shape)
        print(des2.shape)
        matches = bf.match(des1,des2)
        #matches = sorted(matches, key = lambda x: x.distance)   # Sort matches by distance.  Best come first.

        if print_debug == True :
            print('type(matches) : '), print(type(matches))
            print('shape(matches) : '), print(len(matches))
            print(matches[0]),print(matches[1]),print(matches[2]),print(matches[3])
            print(matches[0].queryIdx)
            print(matches[0].trainIdx)
            print(matches[0].distance)

        return matches
            
    elif check == 'without_Lowe_ratio_test' and matching_distance=='NORM_HAMMING':

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck_bool)
        matches = bf.match(des1,des2)

        if print_debug == True :
            print('type(matches) : '), print(type(matches))
            print('shape(matches) : '), print(len(matches))
            print(matches[0]),print(matches[1]),print(matches[2]),print(matches[3])
            print(matches[0].queryIdx)
            print(matches[0].trainIdx)
            print(matches[0].distance)

        return matches
    
    elif check == 'Lowe_ratio_test' and matching_distance=='L2':
    
        print('check: {}'.format(check))
        print('matching_distance: {}'.format(matching_distance))
        print('matching_strategy: {}'.format(matching_strategy))
        print('ratio_thresh: {}'.format(ratio_thresh))

        bf = cv2.BFMatcher(cv2.NORM_L2, False)

        # Ratio Test
        def ratio_test(matches, ratio_thresh):
            prefiltred_matches = []
            for m,n in matches:
                #print('m={} n={}'.format(m,n))
                if m.distance < ratio_thresh * n.distance:
                    prefiltred_matches.append(m)
            return prefiltred_matches
        
        if matching_strategy == 'unidirectional':
            matches01 = bf.knnMatch(des1,des2,k=2)
            good_matches01 = ratio_test(matches01, ratio_thresh)
            return good_matches01
            
        elif matching_strategy == 'intersection':
            matches01 = bf.knnMatch(des1,des2,k=2)
            matches10 = bf.knnMatch(des2,des1,k=2)
            good_matches01 = ratio_test(matches01, ratio_thresh)
            good_matches10 = ratio_test(matches10, ratio_thresh)
            good_matches10_ = {(m.trainIdx, m.queryIdx) for m in good_matches10}
            prefiltred_matches = [m for m in good_matches01 if (m.queryIdx, m.trainIdx) in good_matches10_]
            return prefiltred_matches
            
        elif matching_strategy == 'union':
            matches01 = bf.knnMatch(des1,des2,k=2)
            matches10 = bf.knnMatch(des2,des1,k=2)
            good_matches01 = ratio_test(matches01, ratio_thresh)
            good_matches10 = ratio_test(matches10, ratio_thresh)
            good_matches10_ = {(m.trainIdx, m.queryIdx) for m in good_matches10}
            other_matches = [m for m in good_matches01 if not (m.queryIdx, m.trainIdx) in good_matches10_]
            for m in good_matches10: # added 01/10/2022 
                query = m.queryIdx; train = m.trainIdx # added 01/10/2022
                m.trainIdx = query # added 01/10/2022
                m.queryIdx = train # added 01/10/2022
            prefiltred_matches = good_matches10 + other_matches
            return prefiltred_matches
            
    elif check == 'Lowe_ratio_test' and matching_distance=='NORM_HAMMING':
    
        print('check: {}'.format(check))
        print('matching_distance: {}'.format(matching_distance))
        print('matching_strategy: {}'.format(matching_strategy))
        print('ratio_thresh: {}'.format(ratio_thresh))
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, False)

        # Ratio Test
        def ratio_test(matches, ratio_thresh):
            prefiltred_matches = []
            for m,n in matches:
                #print('m={} n={}'.format(m,n))
                if m.distance < ratio_thresh * n.distance:
                    prefiltred_matches.append(m)
            return prefiltred_matches
        
        if matching_strategy == 'unidirectional':
            matches01 = bf.knnMatch(des1,des2,k=2)
            good_matches01 = ratio_test(matches01, ratio_thresh)
            return good_matches01
            
        elif matching_strategy == 'intersection':
            matches01 = bf.knnMatch(des1,des2,k=2)
            matches10 = bf.knnMatch(des2,des1,k=2)
            good_matches01 = ratio_test(matches01, ratio_thresh)
            good_matches10 = ratio_test(matches10, ratio_thresh)
            good_matches10_ = {(m.trainIdx, m.queryIdx) for m in good_matches10}
            prefiltred_matches = [m for m in good_matches01 if (m.queryIdx, m.trainIdx) in good_matches10_]
            return prefiltred_matches
            
        elif matching_strategy == 'union':
            matches01 = bf.knnMatch(des1,des2,k=2)
            matches10 = bf.knnMatch(des2,des1,k=2)
            good_matches01 = ratio_test(matches01, ratio_thresh)
            good_matches10 = ratio_test(matches10, ratio_thresh)
            good_matches10_ = {(m.trainIdx, m.queryIdx) for m in good_matches10}
            other_matches = [m for m in good_matches01 if not (m.queryIdx, m.trainIdx) in good_matches10_]
            for m in good_matches10: # added 01/10/2022 
                query = m.queryIdx; train = m.trainIdx # added 01/10/2022
                m.trainIdx = query # added 01/10/2022
                m.queryIdx = train # added 01/10/2022
            prefiltred_matches = good_matches10 + other_matches
            return prefiltred_matches