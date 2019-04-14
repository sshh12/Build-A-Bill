import requests
import os


BILL_LIST_URL = "https://www.govtrack.us/api/v2/bill?congress={}"
BILL_TEXT_URL = "https://www.govinfo.gov/content/pkg/{0}/html/{0}.htm"
BAR = '_______________________________________________________________________'


def download_bills(congress):

    bill_list = requests.get(BILL_LIST_URL.format(congress)).json()

    for bill in bill_list["objects"]:

        try:

            name = bill["display_number"]
            bill_url = bill["text_info"]["gpo_url"]
            bill_id = bill_url.split('/')[-1]

            html_bill = requests.get(BILL_TEXT_URL.format(bill_id)).text

            # Clean the raw text
            html_bill = html_bill.replace('<html><body><pre>', '')
            html_bill = html_bill.replace('</pre></body></html>', '')
            html_bill = html_bill.replace('&lt;DOC&gt;', '')
            html_bill = html_bill.replace('&lt;all&gt;', '')
            html_bill = html_bill.split(BAR)[2]
            html_bill = html_bill.strip()

            with open(os.path.join('bills', name + '.txt'), 'w') as bill_file:
                bill_file.write(html_bill)

            print(bill["title"])

        except Exception as e:
            print(e)


if __name__ == "__main__":
    os.makedirs('bills', exist_ok=True)
    # download lots of bills from Congress #100-#116
    for i in range(100, 117):
        download_bills(i)
