# 【3100-312F】 Bopomofo 注音符號
# 【0400-04FF】 Cyrillic 西里爾字母 # 1050

# 【2150-218F】 Number Forms 數字形式
# 【2460-24FF】 Enclosed Alphanumerics 括號字母數字 # 9360
# 【4e00-9faf】 中文
# 【ac00-d7a3】 韓文
# 【f900-fa6a】 中文2

mapHexes = {
    0x00c0: 'a',
    0x00c1: 'a',
    0x00c2: 'a',
    0x00c3: 'a',
    0x00c4: 'a',
    0x00c5: 'a',
    0x00c6: 'ae',
    0x00c7: 'c',
    0x00c8: 'e',
    0x00c9: 'e',
    0x00ca: 'e',
    0x00cb: 'e',
    0x00cc: 'i',
    0x00cd: 'i',
    0x00ce: 'i',
    0x00cf: 'i',
    0x00d0: 'd',
    0x00d1: 'n',
    0x00d2: 'o',
    0x00d3: 'o',
    0x00d4: 'o',
    0x00d5: 'o',
    0x00d6: 'o',
    0x00d7: 'x',
    0x00d8: 'o',
    0x00d9: 'u',
    0x00da: 'u',
    0x00db: 'u',
    0x00dc: 'u',
    0x00dd: 'y',
    0x00de: 'p',
    0x00df: 'b',
    0x00e0: 'a',
    0x00e1: 'a',
    0x00e2: 'a',
    0x00e3: 'a',
    0x00e4: 'a',
    0x00e5: 'a',
    0x00e6: 'ae',
    0x00e7: 'c',
    0x00e8: 'e',
    0x00e9: 'e',
    0x00ea: 'e',
    0x00eb: 'e',
    0x00ec: 'i',
    0x00ed: 'i',
    0x00ee: 'i',
    0x00ef: 'i',
    0x00f0: 'o',
    0x00f1: 'n',
    0x00f2: 'o',
    0x00f3: 'o',
    0x00f4: 'o',
    0x00f5: 'o',
    0x00f6: 'o',
    0x00f8: 'o',
    0x00f9: 'u',
    0x00fa: 'u',
    0x00fb: 'u',
    0x00fc: 'u',
    0x00fd: 'y',
    0x00fe: 'p',
    0x00ff: 'y',

    0x0391: 'a',
    0x0392: 'b',
    0x0395: 'e',
    0x0396: 'z',
    0x0397: 'h',
    0x0399: 'i',
    0x039a: 'k',
    0x039c: 'm',
    0x039d: 'n',
    0x039f: 'o',
    0x03a1: 'p',
    0x03a4: 't',
    0x03a5: 'y',
    0x03a7: 'x',
    0x03ab: 'y',
    0x0410: 'a',
    0x0412: 'b',
    0x041a: 'k',
    0x041c: 'm',
    0x041d: 'h',
    0x041e: 'o',
    0x0420: 'p',
    0x0421: 'c',
    0x0422: 't',
    0x0423: 'y',
    0x0425: 'x',

    0x2160: '1',
    0x2161: '2',
    0x2162: '3',
    0x2163: '4',
    0x2164: '5',
    0x2165: '6',
    0x2166: '7',
    0x2167: '8',
    0x2168: '9',
    0x2169: '10',
    0x216a: '11',
    0x216b: '12',
    0x2170: '1',
    0x2171: '2',
    0x2172: '3',
    0x2173: '4',
    0x2174: '5',
    0x2175: '6',
    0x2176: '7',
    0x2177: '8',
    0x2178: '9',
    0x2179: '10',
    0x217a: '11',
    0x217b: '12',

    0x2460: '1',
    0x2461: '2',
    0x2462: '3',
    0x2463: '4',
    0x2464: '5',
    0x2465: '6',
    0x2466: '7',
    0x2467: '8',
    0x2468: '9',
    0x2469: '10',
    0x246a: '11',
    0x246b: '12',
    0x246c: '13',
    0x246d: '14',
    0x246e: '15',
    0x246f: '16',
    0x2470: '17',
    0x2471: '18',
    0x2472: '19',
    0x2473: '20',
    0x2474: '1',
    0x2475: '2',
    0x2476: '3',
    0x2477: '4',
    0x2478: '5',
    0x2479: '6',
    0x247a: '7',
    0x247b: '8',
    0x247c: '9',
    0x247d: '10',
    0x247e: '11',
    0x247f: '12',
    0x2480: '13',
    0x2481: '14',
    0x2482: '15',
    0x2483: '16',
    0x2484: '17',
    0x2485: '18',
    0x2486: '19',
    0x2487: '20',
    0x2488: '1',
    0x2489: '2',
    0x248a: '3',
    0x248b: '4',
    0x248c: '5',
    0x248d: '6',
    0x248e: '7',
    0x248f: '8',
    0x2490: '9',
    0x2491: '10',
    0x2492: '11',
    0x2493: '12',
    0x2494: '13',
    0x2495: '14',
    0x2496: '15',
    0x2497: '16',
    0x2498: '17',
    0x2499: '18',
    0x249a: '19',
    0x249b: '20',

    0x249c: 'a',
    0x249d: 'b',
    0x249e: 'c',
    0x249f: 'd',
    0x24a0: 'e',
    0x24a1: 'f',
    0x24a2: 'g',
    0x24a3: 'h',
    0x24a4: 'i',
    0x24a5: 'j',
    0x24a6: 'k',
    0x24a7: 'l',
    0x24a8: 'm',
    0x24a9: 'n',
    0x24aa: 'o',
    0x24ab: 'p',
    0x24ac: 'q',
    0x24ad: 'r',
    0x24ae: 's',
    0x24af: 't',
    0x24b0: 'u',
    0x24b1: 'v',
    0x24b2: 'w',
    0x24b3: 'x',
    0x24b4: 'y',
    0x24b5: 'z',
    0x24b6: 'a',
    0x24b7: 'b',
    0x24b8: 'c',
    0x24b9: 'd',
    0x24ba: 'e',
    0x24bb: 'f',
    0x24bc: 'g',
    0x24bd: 'h',
    0x24be: 'i',
    0x24bf: 'j',
    0x24c0: 'k',
    0x24c1: 'l',
    0x24c2: 'm',
    0x24c3: 'n',
    0x24c4: 'o',
    0x24c5: 'p',
    0x24c6: 'q',
    0x24c7: 'r',
    0x24c8: 's',
    0x24c9: 't',
    0x24ca: 'u',
    0x24cb: 'v',
    0x24cc: 'w',
    0x24cd: 'x',
    0x24ce: 'y',
    0x24cf: 'z',
    0x24d0: 'a',
    0x24d1: 'b',
    0x24d2: 'c',
    0x24d3: 'd',
    0x24d4: 'e',
    0x24d5: 'f',
    0x24d6: 'g',
    0x24d7: 'h',
    0x24d8: 'i',
    0x24d9: 'j',
    0x24da: 'k',
    0x24db: 'l',
    0x24dc: 'm',
    0x24dd: 'n',
    0x24de: 'o',
    0x24df: 'p',
    0x24e0: 'q',
    0x24e1: 'r',
    0x24e2: 's',
    0x24e3: 't',
    0x24e4: 'u',
    0x24e5: 'v',
    0x24e6: 'w',
    0x24e7: 'x',
    0x24e8: 'y',
    0x24e9: 'z',

    0x24ea: '0',
    0x24eb: '11',
    0x24ec: '12',
    0x24ed: '13',
    0x24ee: '14',
    0x24ef: '15',
    0x24f0: '16',
    0x24f1: '17',
    0x24f2: '18',
    0x24f3: '19',
    0x24f4: '20',
    0x24f5: '1',
    0x24f6: '2',
    0x24f7: '3',
    0x24f8: '4',
    0x24f9: '5',
    0x24fa: '6',
    0x24fb: '7',
    0x24fc: '8',
    0x24fd: '9',
    0x24fe: '10',

    0x3192: '1',
    0x3193: '2',
    0x3194: '3',
    0x3195: '4',

    0x3220: '1',
    0x3221: '2',
    0x3222: '3',
    0x3223: '4',
    0x3224: '5',
    0x3225: '6',
    0x3226: '7',
    0x3227: '8',
    0x3228: '9',
    0x3229: '10',
    0x322f: '11',
    0x3250: '20',
    0x3251: '21',
    0x3252: '22',
    0x3253: '23',
    0x3254: '24',
    0x3255: '25',
    0x3256: '26',
    0x3257: '27',
    0x3258: '28',
    0x3259: '29',
    0x325a: '30',
    0x325b: '31',
    0x325c: '32',
    0x325d: '33',
    0x325e: '34',
    0x325f: '35',
    0x32b1: '36',
    0x32b2: '37',
    0x32b3: '38',
    0x32b4: '39',
    0x32b5: '40',
    0x32b6: '41',
    0x32b7: '42',
    0x32b8: '43',
    0x32b9: '44',
    0x32ba: '45',
    0x32bb: '46',
    0x32bc: '47',
    0x32bd: '48',
    0x32be: '49',
    0x32bf: '50',

    0xff10: '0',
    0xff11: '1',
    0xff12: '2',
    0xff13: '3',
    0xff14: '4',
    0xff15: '5',
    0xff16: '6',
    0xff17: '7',
    0xff18: '8',
    0xff19: '9',
    0xff21: 'a',
    0xff22: 'b',
    0xff23: 'c',
    0xff24: 'd',
    0xff25: 'e',
    0xff26: 'f',
    0xff27: 'g',
    0xff28: 'h',
    0xff29: 'i',
    0xff2a: 'j',
    0xff2b: 'k',
    0xff2c: 'l',
    0xff2d: 'm',
    0xff2e: 'n',
    0xff2f: 'o',
    0xff30: 'p',
    0xff31: 'q',
    0xff32: 'r',
    0xff33: 's',
    0xff34: 't',
    0xff35: 'u',
    0xff36: 'v',
    0xff37: 'w',
    0xff38: 'x',
    0xff39: 'y',
    0xff3a: 'z',

    0xff41: 'a',
    0xff42: 'b',
    0xff43: 'c',
    0xff44: 'd',
    0xff45: 'e',
    0xff46: 'f',
    0xff47: 'g',
    0xff48: 'h',
    0xff49: 'i',
    0xff4a: 'j',
    0xff4b: 'k',
    0xff4c: 'l',
    0xff4d: 'm',
    0xff4e: 'n',
    0xff4f: 'o',
    0xff50: 'p',
    0xff51: 'q',
    0xff52: 'r',
    0xff53: 's',
    0xff54: 't',
    0xff55: 'u',
    0xff56: 'v',
    0xff57: 'w',
    0xff58: 'x',
    0xff59: 'y',
    0xff5a: 'z',


    0x222a: 'u',
    0x2228: 'v',
    0x03ba: 'k',
    0x0431: '6',

}