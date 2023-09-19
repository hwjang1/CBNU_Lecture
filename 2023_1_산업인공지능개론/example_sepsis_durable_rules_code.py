from durable.lang import *

with statechart('sofa_score'):
    #혈액 관련 수치
    with state('heart'):
        @to('respirationa')
        @when_all(m.bp_level >= 70)
        def heart_calc(c):
            c.s.heart_score = 0

        @to('respirationa')
        @when_all(m.bp_level < 70)
        def heart_calc(c):
            c.s.heart_score = 1

        @to('respirationa')
        @when_all(m.dopamine <= 5)
        def heart_calc(c):
            c.s.heart_score = 2

        @to('respirationa')
        @when_all((m.dopamine > 5) | (m.epinephrine <= 0.1) | (m.norepinephrine <= 0.1))
        def heart_calc(c):
            c.s.heart_score = 3

        @to('respirationa')
        @when_all((m.dopamine > 15) | (m.epinephrine > 0.1) | (m.norepinephrine > 0.1))
        def heart_calc(c):
            c.s.heart_score = 4

    #호흡관련 수치
    with state('respirationa'):
        @to('liver')
        @when_all(m.pao2_fio2 >= 400)
        def pao2_fio2_400(c):
            c.s.respirationa_score = 0

        @to('liver')
        @when_all((m.pao2_fio2 < 400) & (m.pao2_fio2 >= 300))
        def pao2_fio2_300(c):
            c.s.respirationa_score = 1

        @to('liver')
        @when_all((m.pao2_fio2 < 300) & (m.pao2_fio2 >= 200))
        def pao2_fio2_200(c):
            c.s.respirationa_score = 2

        @to('liver')
        @when_all((m.pao2_fio2 < 200) & (m.pao2_fio2 >= 100))
        def pao2_fio2_100(c):
            c.s.respirationa_score = 3

        @to('liver')
        @when_all(m.pao2_fio2 < 100)
        def pao2_fio2_below_100(c):
            c.s.respirationa_score = 4

    #간에 대한 규칙
    with state('liver'):
        @to('platelets')
        @when_all(m.bilirubin_level < 1.2)
        def bilirubin_1_2(c):
            c.s.liver_score = 0


        @to('platelets')
        @when_all((m.bilirubin_level >= 1.2) & (m.bilirubin_level < 2.0))
        def bilirubin_2(c):
            c.s.liver_score = 1


        @to('platelets')
        @when_all((m.bilirubin_level >= 2.0) & (m.bilirubin_level < 6.0))
        def bilirubin_6(c):
            c.s.liver_score = 2


        @to('platelets')
        @when_all((m.bilirubin_level >= 6.0) & (m.bilirubin_level < 12.0))
        def bilirubin_12(c):
            c.s.liver_score = 3


        @to('platelets')
        @when_all(m.bilirubin_level >= 12.0)
        def bilirubin_above_12(c):
            c.s.liver_score = 4

    # 혈약응고에 대한 규칙
    with state('platelets'):
        @to('kidneys')
        @when_all(m.platelet_count >= 150)
        def platelet_150(c):
            c.s.platelets_score = 0


        @to('kidneys')
        @when_all((m.platelet_count < 150) & (m.platelet_count >= 100))
        def platelet_100(c):
            c.s.platelets_score = 1


        @to('kidneys')
        @when_all((m.platelet_count < 100) & (m.platelet_count >= 50))
        def platelet_50(c):
            c.s.platelets_score = 2


        @to('kidneys')
        @when_all((m.platelet_count < 50) & (m.platelet_count >= 20))
        def platelet_20(c):
            c.s.platelets_score = 3


        @to('kidneys')
        @when_all(m.platelet_count < 20)
        def platelet_below_20(c):
            c.s.platelets_score = 4

    # 신장에 대한 규칙
    with state('kidneys'):
        @to('cns')
        @when_all(m.creatinine_level < 1.2)
        def creatinine_1_2(c):
            c.s.kidneys_score = 0


        @to('cns')
        @when_all((m.creatinine_level >= 1.2) & (m.creatinine_level < 2.0))
        def creatinine_2(c):
            c.s.kidneys_score = 1


        @to('cns')
        @when_all((m.creatinine_level >= 2.0) & (m.creatinine_level < 3.5))
        def creatinine_3_5(c):
            c.s.kidneys_score = 2


        @to('cns')
        @when_all(((m.creatinine_level >= 3.5) & (m.creatinine_level < 5.0)) | (m.urine_day < 500))
        def creatinine_5(c):
            c.s.kidneys_score = 3


        @to('cns')
        @when_all((m.creatinine_level >= 5.0) | (m.urine_day < 200))
        def creatinine_above_5(c):
            c.s.kidneys_score = 4

    # 중추신경계에 대한 규칙
    with state('cns'):
        @to('calc')
        @when_all(m.gcs == 15)
        def gcs_15(c):
            c.s.cns_score = 0

        @to('calc')
        @when_all((m.gcs <= 14) & (m.gcs >= 13))
        def gcs_13(c):
            c.s.cns_score = 1

        @to('calc')
        @when_all((m.gcs <= 12) & (m.gcs >= 10))
        def gcs_10(c):
            c.s.cns_score = 2

        @to('calc')
        @when_all((m.gcs <= 9) & (m.gcs >= 6))
        def gcs_6(c):
            c.s.cns_score = 3

        @to('calc')
        @when_all(m.gcs <= 5)
        def gcs_below_6(c):
            c.s.cns_score = 4

    #최종 SOFA 점수 계산
    with state('calc'):
        @to('end')
        @when_all(True)
        def total_score(c):
            c.s.sofa_score = c.s.heart_score \
                            + c.s.respirationa_score \
                            + c.s.liver_score \
                            + c.s.platelets_score \
                            + c.s.kidneys_score \
                            + c.s.cns_score
            print('SOFA 점수는 ', c.s.sofa_score, '입니다.')

    state('end')

assert_fact('sofa_score', {'bp_level': 50,
                           'dopamine': 5,
                           'epinephrine': 0.1,
                           'norepinephrine': 0.1,
                           'pao2_fio2': 100,
                           'bilirubin_level': 1.2,
                           'platelet_count': 100,
                           'creatinine_level': 3.0,
                           'urine_day': 1200,
                           'gcs': 10
                           })
