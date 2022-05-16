def get_psnr(img, largest_pixel_list, smallest_pixel_list, co_largest_pixel_list, co_smallest_pixel_list, max_neighbor, min_neighbor, threshold, chunk_size=2):
    ep = []
    embed_index_max = []
    embed_index_min = []

    contexts = []
    for mn, clpl in zip(max_neighbor, co_largest_pixel_list):
        contexts.append([])
        for xy, pixel in mn:
            if pixel < clpl[1]:
                continue
            contexts[-1].append(((xy[0], xy[1]), pixel))
    P_max = get_predictive_value(contexts)

    for predictive_value, lst1, lst2 in zip(P_max, co_largest_pixel_list, largest_pixel_list):
        c = abs(predictive_value - lst1[1])

        if c < threshold:
            ep.append(lst2[1] - lst1[1])
            embed_index_max.append(lst2[0])

    contexts = []
    for mn, cspl in zip(min_neighbor, co_smallest_pixel_list):
        contexts.append([])
        for xy, pixel in mn:
            if pixel > cspl[1]:
                continue
            contexts[-1].append(((xy[0], xy[1]), pixel))
    P_min = get_predictive_value(contexts)

    for predictive_value, lst1, lst2 in zip(P_min, co_smallest_pixel_list, smallest_pixel_list):
        c = abs(predictive_value - lst1[1])

        if c < threshold:
            ep.append(abs(lst2[1] - lst1[1]))
            embed_index_min.append(lst2[0])

    mc = ep.count(1)
    print(f"the proposed scheme: T={threshold}")
    print(f"Maximum Capacities MC: {mc}")

    print(f"{'embed size':>8s} \t {'PSNR':>10s}")
    rows, cols = get_rows_cols(img)
    tang_three_psnr = []
    for x in np.linspace(0, mc, 20):
        cm = 0
        cn = 0

        cc = 0
        count = x//1

        if count == 0:
            continue
        for _ in ep:
            if _ == 1:
                cc += 1
            else:
                cm += 1

            if cc == count:
                break

        # MSE = ((cm+cn)+(x//2))/((rows - rows %
                # chunk_size)*(cols - cols % chunk_size))
        MSE = (cm+cn)/((rows - rows % chunk_size)*(cols - cols % chunk_size))
        PSNR = 20 * log(255/sqrt(MSE), 10)
        print(f"({x:>.2f}, {PSNR:10f}),")

        tang_three_psnr.append((x, PSNR))
    print()
    return mc, tang_three_psnr, ep, embed_index_max, embed_index_min

def get_predictive_value(contexts):
    predictive_value = []
    for context in contexts:
        A = []
        B = []
        four_neighbors = get_four_neighbor(context, img)
        for index, four_neighbor in enumerate(four_neighbors):
            if four_neighbor.count(0) > 2:
                B.append(index)
            else:
                A.append(index)
        m = len(A) + len(B)
        n = len(B)

        W = [0 for _ in range(m)]

        for index in A:
            W[index] = (1/(get_F(*four_neighbors[index]) + 1)) * ((m-n)/m)
        for index in B:
            W[index] = 1/m

        p_max = 0
        for w, x in zip(W, context):
            p_max += w*x[1]
        predictive_value.append(p_max)
    return predictive_value

def get_four_neighbor(lst: list, img):
    '''
    Returns the 4 neighbors of the image pixel
    '''
    four_neighbor = []
    rows, cols = get_rows_cols(img)

    for index in range(len(lst)):
        it_index, it = lst[index]
        x = it_index[0]
        y = it_index[1]

        left = x-1
        right = x+1
        top = y-1
        bottom = y+1

        if left < 0:
            left = -1
        if top < 0:
            top = -1
        if right >= cols:
            right = -1
        if bottom >= rows:
            bottom = -1

        if top != -1:
            a = img[x, top]
        else:
            a = 0

        if left != -1:
            c = img[left, y]
        else:
            c = 0

        if right != -1:
            d = img[right, y]
        else:
            d = 0

        if bottom != -1:
            b = img[x, bottom]
        else:
            b = 0

        four_neighbor.append([a, b, c, d])

    return four_neighbor

def get_rows_cols(img):
    '''
    Returns the number of rows and columns of the image
    '''
    rows, cols = img.shape
    if rows > cols:
        rows = cols
    else:
        cols = rows

    return rows, cols

def pixel_sum(x,y):
    if type(x) is int:
        return x+y
    
    if type(x) is tuple:
        return tuple([i+y for i in x])

def show_embed_img(img, img_path, embed_index_max, embed_index_min):  # 80
    img_embed = Image.open(img_path)
    for x, y in embed_index_max:
        if img[x, y] != 255:
            img_embed.putpixel((x,y), pixel_sum(img_embed.getpixel((x,y)), random.randint(0,1)))

    for x, y in embed_index_min:
        if img[x, y] != 255:
            img_embed.putpixel((x,y), pixel_sum(img_embed.getpixel((x,y)), random.randint(0,1)))
    img_embed.save(f"{img_path.split('.')[0]}_embed.{img_path.split('.')[-1]}")
    print(f"Save {img_path.split('.')[0]}_embed.{img_path.split('.')[-1]}")
