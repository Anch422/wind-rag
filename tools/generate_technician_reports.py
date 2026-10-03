"""Create 45 separate searchable technician reports for CARE anomalies.

Run with a Python environment containing pandas and reportlab. These are
generated research fixtures, not verified CARE diagnoses.
"""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from html import escape
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from windrag.data import load_corpus
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'datasets' / 'technician_reports'
COMPANIES = ['Northwind Field Services', 'AeroPeak Maintenance', 'Harbor Turbine Care',
             'GreenSpan Engineering', 'Summit Rotor Services', 'BlueVale Renewables']
TECHNICIANS = ['Nora Chen', 'Mateo Silva', 'Leah Brooks', 'Arun Patel', 'Mina Park', 'Jonas Reed',
               'Sofia Lin', 'Theo Martin', 'Elena Cruz', 'Owen Hale', 'Priya Shah', 'Kai Morgan']

# Matching selects a fictional scenario only; it does not establish a real cause.
SCENARIOS = [
    (['overpressure'], 'Transformer pressure monitoring assembly', 'A drifting pressure switch and restricted breather were observed in the mock inspection.', 'Restricted ventilation or a pressure-switch fault may explain the reported indication.', 'Replace the faulty pressure switch and service the breather using the approved transformer work instruction.'),
    (['transformer'], 'Transformer cooling fan and intake filter', 'The mock inspection found a stalled cooling fan and a dust-loaded intake filter.', 'Reduced cooling airflow is a suspected contributor; internal transformer damage was not established.', 'Replace the cooling fan assembly and renew the intake filter; inspect the cooling path under the approved service plan.'),
    (['generator bearing'], 'Generator drive-end bearing', 'The fictional teardown noted bearing-race surface damage and discolored lubricant.', 'Lubrication degradation or misalignment may have contributed; no real bearing analysis was performed.', 'Replace the affected bearing assembly and lubricant; verify generator alignment against the applicable manufacturer procedure.'),
    (['rotorbearing', 'rotor bearing', 'main bearing'], 'Main rotor bearing assembly', 'The mock bearing inspection recorded raceway wear and abnormal rotation resistance.', 'Wear progression or uneven load distribution is suspected; the initiating mechanism remains uncertain.', 'Replace the damaged main-bearing assembly under an approved lifting and alignment plan; inspect adjacent supports.'),
    (['coupling'], 'Gear-oil pump motor coupling', 'The fictional inspection identified cracking in the flexible coupling element.', 'Coupling fatigue or pump-to-motor misalignment is a possible cause.', 'Replace the damaged coupling element and check pump/motor alignment before the approved return-to-service tests.'),
    (['gearbox', 'gear oil supply', 'oil leakage'], 'Gearbox lubrication pump and bearing module', 'The mock inspection noted degraded lubrication delivery and wear in the affected bearing module.', 'Restricted oil delivery or progressive bearing wear may have contributed; the diagnosis is synthetic.', 'Service the lubrication circuit and replace the affected pump or bearing module identified in the fictional inspection; renew filters and lubricant as specified by the service plan.'),
    (['harting', 'wiring blade'], 'Hub blade-control connector and harness', 'The fictional inspection recorded damaged connector contacts and a worn harness strain relief.', 'Vibration-related connector wear or moisture exposure is suspected.', 'Replace the damaged connector and harness section; restore strain relief and verify blade-control communication using the approved test procedure.'),
    (['fuse filter'], 'Converter filter-supply module', 'The mock inspection identified an open filter-supply fuse and a defective supply module.', 'A supply-module fault may have overloaded the protective fuse; the initiating fault was not verified.', 'Replace the defective filter-supply module and specified protective fuse after the fault review; perform the prescribed converter functional test.'),
    (['fan on pitch'], 'Pitch-motor cooling fan', 'The fictional inspection recorded a stalled fan and obstruction in the fan housing.', 'Reduced cooling or fan-bearing wear is the suspected contributor.', 'Replace the pitch-motor fan and clear the cooling path; complete the manufacturer-specified pitch-axis functional checks.'),
    (['encoder'], 'Pitch-position encoder', 'The mock inspection recorded intermittent position feedback and a damaged encoder connector.', 'Encoder or connector deterioration may explain intermittent pitch-position feedback.', 'Replace the faulty encoder/connector assembly and recalibrate feedback using the approved pitch-controller procedure.'),
    (['beckhoffcard', 'beckhoff card'], 'Pitch-controller I/O card', 'The fictional inspection identified intermittent operation in the pitch-controller I/O card.', 'An electronic card fault or connector deterioration is suspected; the real event was not diagnosed.', 'Replace the faulty I/O card with an approved compatible module and restore the controlled configuration; verify pitch-axis feedback.'),
    (['communication fault', 'bk1120'], 'Controller communication coupler', 'The mock inspection found an intermittent coupler connection and a damaged communication cable.', 'Coupler degradation or cable damage may explain communication loss.', 'Replace the faulty coupler and damaged cable; restore the approved controller configuration and perform network communication checks.'),
    (['slip ring', 'carbonbrush'], 'Hub slip-ring brush module', 'The fictional inspection recorded worn brushes and contamination on the contact path.', 'Brush wear or poor contact pressure is a possible contributor to intermittent signals.', 'Replace the worn brush module and service the contact path under the approved slip-ring procedure; verify hub communication.'),
    (['battery charger'], 'Hub battery charger', 'The mock inspection identified an intermittent charger output fault.', 'Charger electronics degradation is suspected; the battery condition requires separate verification.', 'Replace the faulty charger and check battery condition with the prescribed diagnostic procedure; verify the approved backup-power test.'),
    (['battery', 'batterien', 'dc-link', 'batt', 'rewiring'], 'Pitch backup battery and wiring harness', 'The fictional inspection recorded a degraded battery module and loose harness termination.', 'Battery aging or intermittent wiring contact may explain backup-power faults.', 'Replace the degraded battery module and repair the identified harness termination under the approved service procedure; run the backup-power verification.'),
    (['water cooling', 'wrong position'], 'Water-cooling isolation valve', 'The mock inspection found the cooling valve in a position inconsistent with the approved operating configuration.', 'An incomplete maintenance handover is a possible contributor; this is a fictional process finding.', 'Restore the valve to the approved operating configuration and review the maintenance handover checklist; verify coolant circulation.'),
    (['grease pump'], 'Yaw lubrication pump', 'The fictional inspection identified a seized lubrication-pump drive and restricted grease delivery.', 'Pump-drive wear or contaminated grease may have contributed.', 'Replace the defective lubrication pump and service the restricted delivery path using the approved lubrication plan.'),
    (['grease collector'], 'Blade grease collector', 'The mock inspection found a missing grease-collector assembly at blade 3.', 'Incomplete reassembly after maintenance is a possible cause.', 'Install the approved replacement grease collector and verify its mounting and collection path against the service checklist.'),
    (['rcd'], 'Cabinet residual-current protection device', 'The fictional inspection recorded failure of the protective device during the prescribed inspection.', 'Protective-device deterioration is suspected; circuit condition requires qualified assessment.', 'Replace the faulty protective device with the approved rated unit and complete the qualified electrical verification record.'),
    (['cooler bypass'], 'Gear-oil cooler bypass valve', 'The mock inspection recorded a sticking valve and contamination in the bypass path.', 'Valve contamination or actuator wear may have contributed to unstable cooling.', 'Replace or service the sticking bypass-valve assembly under the approved work instruction; verify the lubrication cooling response.'),
    (['24vac', 'rotorbrake', 'rotor brake'], 'Rotor-brake hydraulic control module', 'The fictional inspection recorded intermittent brake-control response and degraded hydraulic-pump delivery.', 'A control-supply interruption or hydraulic-module fault is suspected; multiple possible causes remain.', 'Replace the faulty control or hydraulic module identified in the mock inspection and inspect related connections; complete the prescribed brake-control verification.'),
    (['hydraulic'], 'Hydraulic power-unit pump and accumulator', 'The mock inspection recorded degraded pump delivery and loss of accumulator performance.', 'Pump wear or accumulator deterioration may explain hydraulic-pressure instability.', 'Replace the affected pump and accumulator assembly under the approved depressurization and service plan; verify hydraulic response using the prescribed test.'),
    (['axis 3', 'axis 1', 'axis 2'], 'Pitch-axis drive feedback module', 'The fictional inspection identified an intermittent drive feedback connection.', 'Connector deterioration or a feedback-module fault is suspected.', 'Replace the faulty feedback module and damaged connector, restore the approved drive configuration, and complete pitch-axis verification.'),
    (['current measurement'], 'Auxiliary-current measurement transducer', 'The mock inspection recorded inconsistent transducer readings during the controlled diagnostic check.', 'Transducer drift or a connection fault is a possible contributor.', 'Replace the faulty measurement transducer and repair the identified connection; verify calibration with the approved reference procedure.'),
    (['umrichter', 'converter'], 'Converter feedback and cooling module', 'The fictional inspection recorded an intermittent feedback module and degraded cooling airflow.', 'Feedback instability or thermal stress may have contributed; a causal link was not confirmed.', 'Replace the identified feedback/cooling module and restore the approved converter configuration; complete the specified functional verification.'),
    (['grounding', 'lightning'], 'Hub grounding contact and protective cover', 'The mock inspection recorded worn grounding contact material and a damaged protective cover.', 'Contact wear or incomplete reassembly may have contributed.', 'Replace the worn grounding contact and missing protective cover using approved parts; complete the qualified continuity inspection.'),
]


def scenario(description):
    return next((entry[1:] for entry in SCENARIOS if any(key in description.lower() for key in entry[0])),
                ('Turbine control interface', 'The mock inspection found an intermittent control interface.',
                 'A connector or control-module fault is suspected, but the cause remains uncertain.',
                 'Replace the identified faulty interface and perform the approved functional verification.'))


def report_text(text):
    """Keep provenance in the manifest, rather than repeating it in PDF evidence."""
    import re
    text = text.replace('; the diagnosis is synthetic.', '.')
    text = text.replace('; the real event was not diagnosed.', '; the initiating cause remains unconfirmed.')
    text = text.replace('no real bearing analysis was performed', 'bearing analysis has not confirmed the initiating mechanism')
    return re.sub(r'\b(?:mock|fictional|synthetic)\s+', '', text, flags=re.I)


def generate():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _, events, _, digest = load_corpus(ROOT / 'datasets' / 'CARE_To_Compare')
    anomalies = sorted((e for e in events if e['label'] == 'anomaly'), key=lambda e: (e['farm'], e['start'], e['event_id']))
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle('ReportTitle', fontName='Helvetica-Bold', fontSize=19, leading=24, textColor=colors.HexColor('#12334a'), spaceAfter=10))
    styles.add(ParagraphStyle('SectionTitle', fontName='Helvetica-Bold', fontSize=11, leading=14, textColor=colors.HexColor('#087f78'), spaceBefore=12, spaceAfter=5))
    styles['BodyText'].fontSize = 10
    styles['BodyText'].leading = 14
    records = []
    for farm in ('A', 'B', 'C'):
        farm_events = [e for e in anomalies if e['farm'].endswith(farm)]
        def para(text, style='BodyText'):
            return Paragraph(escape(str(text)), styles[style])
        def section(title, text):
            story.extend([para(title, 'SectionTitle'), para(text)])
        for number, event in enumerate(farm_events):
            story = []
            ordinal = anomalies.index(event)
            report_id = f'TR-{farm}-{int(event["event_id"]):03d}'
            date = datetime.fromisoformat(event['end']) + timedelta(days=1 + ordinal % 7)
            filename = f'{date:%Y-%m-%d}_Wind_Farm_{farm}_Event_{int(event["event_id"]):03d}_{report_id}.pdf'
            part, finding, cause, repair = map(report_text, scenario(event['description']))
            report = {'report_id': report_id, 'farm': event['farm'], 'event_id': event['event_id'], 'asset': event['asset'],
                      'service_date': date.isoformat(timespec='minutes'), 'company': COMPANIES[ordinal % len(COMPANIES)],
                      'technician': TECHNICIANS[ordinal % len(TECHNICIANS)], 'component': part, 'finding': finding,
                      'possible_cause': cause, 'repair': repair, 'pdf': filename,
                      'inspection_page': 1, 'repair_page': 2,
                      'data_origin': 'generated_research_fixture', 'care_event_document_id': event['document_id']}
            records.append(report)
            story.extend([para('Technician service report', 'ReportTitle'), para('Inspection findings, affected component and possible cause')])
            metadata = [('Report ID:', report_id), ('Farm / event:', f'{event["farm"]} / Event {event["event_id"]}'),
                        ('Turbine asset:', event['asset']), ('Service date:', report['service_date']),
                        ('Company:', report['company']), ('Technician:', report['technician']),
                        ('Affected component:', part)]
            table = Table([[para(k), para(v)] for k, v in metadata], colWidths=[115, 365])
            table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#eef6f8')),
                                      ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('TOPPADDING', (0, 0), (-1, -1), 5),
                                      ('BOTTOMPADDING', (0, 0), (-1, -1), 5)]))
            story.extend([Spacer(1, 12), table])
            section('CARE recorded anomaly', event['description'].replace('\ufffd', '?'))
            section('Inspection findings', finding)
            section('Possible cause', cause)
            section('Cause status', 'Suspected and unconfirmed. Additional component inspection and failure analysis are needed to confirm the initiating mechanism.')
            section('Additional notes', 'Retain the event identifier, inspection findings and component service record together for follow-up review.')
            story.append(PageBreak())
            story.extend([para('Repair and verification', 'ReportTitle'), para('Corrective action record and return-to-service checks'),
                          para(f'Report ID: {report_id} | {event["farm"]} | Event {event["event_id"]} | Turbine asset {event["asset"]}'),
                          para(f'Service date: {report["service_date"]} | Company: {report["company"]} | Technician: {report["technician"]}'),
                          para(f'Affected component: {part}')])
            section('Repair record', repair)
            section('Parts and materials', f'Approved replacement {part.lower()} or the affected subassembly, with the matching manufacturer configuration. Confirm part numbers, torque settings and electrical ratings against the applicable manufacturer service documentation.')
            section('Verification', 'The service team completed visual checks, the component-specific functional check and a controlled return-to-service observation. No recurring alarm was noted during the recorded observation window. Continue condition monitoring and review recurring symptoms.')
            section('Safety and limitations', 'Work requires qualified technicians, an approved work order, isolation and access procedures, and the applicable turbine manufacturer service documentation. Use this record with the approved component-specific service procedure.')
            section('Additional notes', 'Keep possible causes distinct from confirmed findings. Record follow-up observations against this report identifier and review the component if symptoms recur.')
            def footer(canvas, doc):
                canvas.setFont('Helvetica', 8)
                canvas.setFillColor(colors.HexColor('#506577'))
                canvas.drawString(48, 26, f'{report_id} | Wind Farm {farm} | Technician service report | Page {doc.page} of 2')
                canvas.setStrokeColor(colors.HexColor('#bdd6df'))
                canvas.line(48, 40, A4[0] - 48, 40)
            SimpleDocTemplate(str(OUTPUT / filename), pagesize=A4, rightMargin=48, leftMargin=48,
                              topMargin=42, bottomMargin=52, title=f'{report_id} - Technician service report', author=report['company']).build(story, onFirstPage=footer, onLaterPages=footer)
    manifest = {'data_origin': 'generated_research_fixture', 'externally_verified': False, 'description': '45 generated technician reports for controlled RAG experiments. Findings and repair outcomes are not independently verified CARE field records.',
                'care_corpus_sha256': digest, 'report_count': len(records), 'reports': records}
    (OUTPUT / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    (OUTPUT / 'README.md').write_text('# Technician PDF dataset\n\n45 individual two-page PDF reports (90 pages). Each report maps to one recorded CARE anomaly and includes a service date, company, technician, affected component, possible cause and repair record.\n\nThe dataset is generated for controlled research. Generation history and verification status are preserved in `manifest.json`, which is not indexed or sent to the LLM. Only the original event identifier/asset/anomaly description comes from CARE.\n\nPDFs are searchable; no OCR is required. Regenerate with `tools/generate_technician_reports.py` in an environment with reportlab and pandas.\n', encoding='utf-8')
    print(json.dumps({'reports': len(records), 'pdfs': len(records), 'pages': len(records) * 2, 'folder': str(OUTPUT)}, indent=2))


if __name__ == '__main__':
    generate()
