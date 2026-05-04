from evaluate import normalise, to_words
import collections

def diff_words(gt_file, ocr_file):
    gt = open(gt_file, encoding='utf-8').read()
    ocr = open(ocr_file, encoding='utf-8').read()
    gc = collections.Counter(sorted(to_words(normalise(gt))))
    oc = collections.Counter(sorted(to_words(normalise(ocr))))
    return list((gc-oc).elements()), list((oc-gc).elements())

print('--- BATCH 2 ERRORS ---')
gt2_del, gt2_ins = diff_words('ground_truth/batch2.txt', 'output/batch2_plain.txt')
print('Missing from OCR (Deletions):', gt2_del)
print('Extra in OCR (Insertions):', gt2_ins)

print('\n--- BATCH 3 ERRORS ---')
gt3_del, gt3_ins = diff_words('ground_truth/batch3.txt', 'output/deepseek_test/batch3-0999.txt')
print('Missing from OCR (Deletions):', gt3_del)
print('Extra in OCR (Insertions):', gt3_ins)
