import os
import json
import pymysql
from datetime import datetime
import argparse

DB_CONFIG = {
    'host': '192.168.250.15',
    'user': 'c4y_portal',
    'password': 'p2rt@1',
    'database': 'ePortaldb'
}

def get_section_id_map(conn):
    # Returns dict mapping section_code to section_id.
    with conn.cursor() as cur:
        cur.execute("SELECT section_id, section_code FROM section")
        return {row[1]: row[0] for row in cur.fetchall()}

def get_inspection_id(conn, store_id, quarter):
    # Returns inspection_id for store and quarter, or None.
    with conn.cursor() as cur:
        cur.execute(
            "SELECT inspection_id FROM inspection WHERE store_id = %s AND quarter_no = %s",
            (store_id, quarter)
        )
        result = cur.fetchone()
        return result[0] if result else None

def get_next_audit_round(conn, inspection_id, section_id):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT MAX(audit_round) FROM submission_audit_automation
            WHERE inspection_id = %s AND section_id = %s
        """, (inspection_id, section_id))
        result = cur.fetchone()
        last_round = result[0] if result and result[0] is not None else 0
        return last_round + 1

def delete_previous_audit(conn, inspection_id, section_id):
    with conn.cursor() as cur:
        cur.execute("""
            DELETE FROM submission_audit_automation
            WHERE inspection_id = %s AND section_id = %s
        """, (inspection_id, section_id))
    conn.commit()

def insert_audit_result(conn, audit):
    # Insert a row into submission_audit_automation.
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO submission_audit_automation
            (inspection_id, section_id, submission_id, auditor_id, audited_at, audit_round, result, comments, actions)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            audit['inspection_id'],
            audit['section_id'],
            audit.get('submission_id', 0),
            audit['auditor_id'],
            audit['audited_at'],
            audit['audit_round'],
            audit['result'],
            audit.get('comments', ""),
            audit.get('actions', "")
        ))
    conn.commit()

def main(store_id, quarter, output_dir, auditor_id, score, month, year):
    try:
        # 1. Build the file path - Use absolute path
        if not os.path.isabs(output_dir):
            PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(PROJECT_ROOT, output_dir)
        file = f"store{store_id:03d}.json"
        path = os.path.join(output_dir, file)
        if not os.path.exists(path):
            print(f"Result file {file} not found in {output_dir}")
            return

        # 2. Open DB connection
        print("Connecting to DB:", DB_CONFIG)
        conn = pymysql.connect(**DB_CONFIG)
        section_map = get_section_id_map(conn)

        # 3. Load audit results from JSON
        with open(path, 'r', encoding='utf-8') as f:
            result_data = json.load(f)

        # 4. Get inspection_id for this store/quarter
        inspection_id = get_inspection_id(conn, store_id, quarter)
        if not inspection_id:
            print(f"No inspection_id found for store {store_id} Q{quarter}, skipping.")
            conn.close()
            return

        # 5. Loop through all sections and insert
        for section_code, section_result in result_data.items():
            section_id = section_map.get(section_code)
            if not section_id:
                print(f"Section code {section_code} not found in DB, skipping.")
                continue
            audit_round = get_next_audit_round(conn, inspection_id, section_id)
            try:
                delete_previous_audit(conn, inspection_id, section_id)
                audit = {
                    'inspection_id': inspection_id,
                    'section_id': section_id,
                    'submission_id': 0,
                    'auditor_id': auditor_id,
                    'audited_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'audit_round': audit_round,
                    'result': 'Pass' if section_result.get('pass') else 'Fail',
                    'comments': "; ".join(section_result.get('visual_feedback', [])),
                    'actions': "; ".join(section_result.get('action_plan', []))
                }
                insert_audit_result(conn, audit)
                print(f"Inserted {store_id} {section_code} round {audit_round} -> {audit['result']}")
            except Exception as section_e:
                print(f"❌ ERROR inserting/updating section {section_code}: {section_e}")

        # === Insert store-level summary ===
        try:
            with conn.cursor() as cur:
                # Remove any existing result for the same inspection_id
                cur.execute("""
                    DELETE FROM submission_audit_result_automation
                    WHERE inspection_id = %s
                """, (inspection_id,))

                # Determine status
                status = "Completed" if score >= 88 else "In Progress"
                e_sign_off = datetime.now().strftime("YXY %m/%d/%Y %I:%M:%S %p")
                e_signoff_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                cur.execute("""
                    INSERT INTO submission_audit_result_automation
                    (inspection_id, score, status, auditor_id, e_sign_off, e_signoff_time)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                    score = VALUES(score),
                    status = VALUES(status),
                    auditor_id = VALUES(auditor_id),
                    e_sign_off = VALUES(e_sign_off),
                    e_signoff_time = VALUES(e_signoff_time)
                """, (
                    inspection_id,
                    score,
                    status,
                    auditor_id,
                    e_sign_off,
                    e_signoff_time
                ))

                conn.commit()
                print(f"✅ Store score {score:.1f}% inserted for inspection {inspection_id}")
        except Exception as store_score_err:
            print(f"❌ Failed to insert store-level score: {store_score_err}")

        # === Insert/Update branch_inspection_automation (AFTER all other DB work, before conn.close) ===
        try:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO branch_inspection_automation
                        (store_id, month, year, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        updated_at = VALUES(updated_at)
                """, (
                    store_id, month, year, now, now
                ))
                conn.commit()
                print(f"✅ Automation record inserted/updated for store {store_id} ({month}/{year})")
        except Exception as e:
            print(f"❌ Failed to insert/update automation tracking: {e}")

        conn.close()
        print("Import finished.")

    except Exception as e:
        print(f"❌ Top-level ERROR in upLoader.py: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--store_id", type=int, required=True)
    parser.add_argument("--quarter", type=int, required=True)
    parser.add_argument("--output_dir", type=str, default='outputs\\o4-mini')
    parser.add_argument("--auditor_id", type=int, default=00000)
    parser.add_argument("--score", type=float, default=0.0)
    parser.add_argument("--month", type=int, required=True, help="Month of automated evaluation (1-12)")
    parser.add_argument("--year", type=int, required=True, help="Year of automated evaluation (e.g., 2025)")
    args = parser.parse_args()

    main(
        store_id=args.store_id,
        quarter=args.quarter,
        output_dir=args.output_dir,
        auditor_id=args.auditor_id,
        score=args.score,
        month=args.month,
        year=args.year
    )
