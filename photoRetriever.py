import os
import threading
import pymysql
import logging
import time
import argparse

from typing import List, Dict

# --- Configuration ---
DB_CONFIG = {
    'host': '192.168.250.15',
    'user': 'c4y_portal',
    'password': 'p2rt@1',
    'database': 'ePortaldb'
}

#UPLOAD_IMAGE_DIR = '/branch_inspection/upload_images'
UPLOAD_IMAGE_DIR = 'C:/Users/Xavier/Desktop/Work/AutomationProject/upload_images'
LOG_FILE = 'logs/photo_retriever.log'
MAX_RETRIES = 3
RETRY_DELAY = 1  # in seconds

# --- Logging Setup ---  Temp Disabled for the sake of cleaning
#logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Photo Retriever Class ---
class PhotoRetriever:
    def __init__(self, store_id: int, quarter: int):
        self.store_id = store_id
        self.quarter = quarter
        self.photos_by_section: Dict[str, List[Dict]] = {}
        self.lock = threading.Lock()

    def fetch_sections(self) -> List[Dict]:
        with pymysql.connect(**DB_CONFIG) as conn:
            with conn.cursor(pymysql.cursors.DictCursor) as cursor:
                cursor.execute("""
                    SELECT DISTINCT s.section_id, s.section_code
                    FROM section_submission ss
                    JOIN section s ON ss.section_id = s.section_id
                    JOIN inspection i ON ss.inspection_id = i.inspection_id
                    WHERE i.store_id = %s AND i.quarter_no = %s
                """, (self.store_id, self.quarter))
                return cursor.fetchall()

    def get_inspection_id(self) -> int:
        with pymysql.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT inspection_id FROM inspection
                    WHERE store_id = %s AND quarter_no = %s
                """, (self.store_id, self.quarter))
                result = cursor.fetchone()
                if result:
                    return result[0]
                raise ValueError("No inspection found for store {} Q{}".format(self.store_id, self.quarter))


    def get_submission_ids(self, conn, inspection_id: int, section_id: int) -> List[int]:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT submission_id FROM section_submission
                WHERE inspection_id = %s AND section_id = %s
            """, (inspection_id, section_id))
            return [row[0] for row in cursor.fetchall()]

    def get_photo_data_for_submission(self, conn, submission_id: int, section_code: str, photo_index_start: int) -> List[Dict]:
        photos = []
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT photo_src FROM submission_photo
                    WHERE submission_id = %s
                """, (submission_id,))
                results = cursor.fetchall()
        except Exception as e:
            logging.error(f"Failed to fetch photo_srcs for store {self.store_id:03d}, section {section_code}, submission {submission_id}: {e}")
            return photos

        for idx, row in enumerate(results):
            photo_src = row[0]
            photo_path = os.path.join(UPLOAD_IMAGE_DIR, photo_src).replace('\\', '/')
            ext = os.path.splitext(photo_src)[1].lower() or '.jpg'
            index = photo_index_start + idx
            filename = f"store{self.store_id:03d}_{section_code}_{index}{ext}"

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    if not os.path.exists(photo_path):
                        raise FileNotFoundError(f"Photo not found at {photo_path}")
                    with open(photo_path, 'rb') as f:
                        image_bytes = f.read()
                    photos.append({
                        'filename': filename,
                        'image_bytes': image_bytes,
                        'original_path': photo_src  # NEW: Save original DB path
                    })
                    break
                except Exception as e:
                    if attempt == MAX_RETRIES:
                        logging.error(f"Failed to load photo after {MAX_RETRIES} attempts: Store {self.store_id:03d}, Section {section_code}, File {filename} — {e}")
                    else:
                        time.sleep(RETRY_DELAY)
        return photos
    
    # -- Returns the commment of a store regarding 'Reason if picture is not standard'
    def get_reason_comment_for_section(self, conn, section_id, submission_id):
        """Fetch the store justification comment for this section submission."""
        with conn.cursor() as cursor:
            # 1. Find field_id for "Reason if picture is not standard"
            cursor.execute("""
                SELECT field_id
                FROM section_field
                WHERE section_id = %s
                AND LOWER(label) = 'reason if picture is not standard'
                LIMIT 1
            """, (section_id,))
            row = cursor.fetchone()
            if not row:
                return ""
            field_id = row[0]
            # 2. Get the comment value
            cursor.execute("""
                SELECT value_text
                FROM submission_field_value
                WHERE submission_id = %s AND field_id = %s
                LIMIT 1
            """, (submission_id, field_id))
            comment_row = cursor.fetchone()
            comment = comment_row[0] if comment_row and comment_row[0] else ""
            return comment, field_id
        
    # --- Return a list of stores that have completed all 25 sections ---
    @classmethod
    def get_stores_with_all_sections(cls, quarter: int, required_sections: int = 25) -> List[int]:
        stores = []
        try:
            with pymysql.connect(**DB_CONFIG) as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT i.store_id
                        FROM inspection i
                        JOIN section_submission ss ON i.inspection_id = ss.inspection_id
                        WHERE i.quarter_no = %s
                        GROUP BY i.store_id
                        HAVING COUNT(DISTINCT ss.section_id) >= %s
                    """, (quarter, required_sections))
                    rows = cursor.fetchall()
                    excluded = {990, 991}
                    stores = [row[0] for row in rows if row[0] not in excluded]

        except Exception as e:
            logging.error(f"Failed to fetch stores with complete section submissions: {e}")
        return stores
    
    def process_section(self, section_code: str, section_id: int, inspection_id: int):
        all_submissions = []
        try:
            with pymysql.connect(**DB_CONFIG) as conn:
                # Only get Pending submissions for this section!
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT submission_id
                        FROM section_submission
                        WHERE inspection_id = %s AND section_id = %s AND status = 'Approved'
                    """, (inspection_id, section_id))
                    submission_ids = [row[0] for row in cursor.fetchall()]

                photo_index = 0
                for sub_id in submission_ids:
                    photos = self.get_photo_data_for_submission(conn, sub_id, section_code, photo_index)
                    store_comment,field_id = self.get_reason_comment_for_section(conn, section_id, sub_id)
                    all_submissions.append({
                        "photos": photos,
                        "store_comment": store_comment,
                        "field_id": field_id,
                        "section_id": section_id,
                        "submission_id": sub_id  # optional, for traceability
                    })
                    photo_index += len(photos)
        except Exception as e:
            logging.error(f"Error processing section {section_code} for store {self.store_id:03d}: {e}")
            return

        with self.lock:
            self.photos_by_section[section_code] = all_submissions


    def retrieve_all_photos(self) -> Dict[str, List[Dict]]:
        inspection_id = self.get_inspection_id()
        sections = self.fetch_sections()

        threads = []
        for section in sections:
            t = threading.Thread(
                target=self.process_section,
                args=(section['section_code'], section['section_id'], inspection_id)
            )
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        return self.photos_by_section

# --- Example Usage ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retrieve branch submission photos by store and quarter.')
    parser.add_argument('--store_id', type=int, required=True, help='Store ID (e.g., 1 for store001)')
    parser.add_argument('--quarter', type=int, required=True, help='Quarter number (1 to 4)')

    args = parser.parse_args()

    retriever = PhotoRetriever(args.store_id, args.quarter)
    result = retriever.retrieve_all_photos()

    logging.info("Retrieved submission photos by section:")
    for section, photos in result.items():
        logging.info(f"Store {args.store_id:03d} - Section {section} has {len(photos)} photos.")
        for photo in photos:
            logging.info(f"- {photo['filename']}")

# --- Helper Function for API ---
def get_failed_section_codes(store_id, quarter):
        DB_CONFIG = {
            'host': '192.168.250.15',
            'user': 'c4y_portal',
            'password': 'p2rt@1',
            'database': 'ePortaldb'
        }
        conn = pymysql.connect(**DB_CONFIG)
        try:
            with conn.cursor() as cur:
                # 1. Get inspection_id for this store and quarter
                cur.execute(
                    "SELECT inspection_id FROM inspection WHERE store_id=%s AND quarter_no=%s",
                    (store_id, quarter)
                )
                row = cur.fetchone()
                if not row:
                    return []
                inspection_id = row[0]

                # 2. Get failed section_ids
                cur.execute(
                    "SELECT section_id FROM submission_audit_automation WHERE inspection_id=%s AND result='Fail'",
                    (inspection_id,)
                )
                failed_section_ids = [r[0] for r in cur.fetchall()]
                if not failed_section_ids:
                    return []

                # 3. Map section_id to section_code
                cur.execute(
                    f"SELECT section_id, section_code FROM section WHERE section_id IN ({','.join(['%s']*len(failed_section_ids))})",
                    tuple(failed_section_ids)
                )
                id_code_map = {row[0]: row[1] for row in cur.fetchall()}
                return [id_code_map[sec_id] for sec_id in failed_section_ids if sec_id in id_code_map]
        finally:
            conn.close()