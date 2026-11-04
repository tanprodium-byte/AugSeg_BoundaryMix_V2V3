"""CPU/standard-library fixtures; never access real inputs or target state."""
import csv, json, os, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import portable as m

def writejson(p,x):p.write_text(json.dumps(x))
def csvfile(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

class PortableChecks(unittest.TestCase):
 def test_01_package_and_source_byte_pins(self):self.assertEqual(m.verify_package()['source_files'],29)
 def test_02_checkpoint_identity_and_validation_threshold_unchanged(self):
  p=m.load('plan.json');old=m.load('historical/plan.json')
  for k in ['teacher_execution_id','expected_best_epoch_zero_based','expected_checkpoint_next_epoch','expected_teacher_miou_percent','validation_abs_tolerance_pp','precision','resource_policy','batch','diagnostic_batches','mix_seed_base']:self.assertEqual(p[k],old[k])
 def test_03_plan_paths_and_transfer_excludes_split_bytes(self):
  self.assertFalse((m.C/'locked_lists.json').exists());self.assertFalse(list(m.C.rglob('*.pth')))
  inp=m.load('inputs.json');self.assertEqual({x['lines'] for x in inp['splits']},{662,1323,2646,1449})
  self.assertTrue(all('/splits/pascal/' in x['target_path'] for x in inp['splits']))
 def payload_fixture(self,td):
  base=Path(td);root=base/'data';rows=[]
  for kind,rel in [('image','JPEGImages/receiver.jpg'),('mask','SegmentationClassAug/receiver.png')]:
   f=root/rel;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(b'synthetic bytes '+kind.encode());rows.append({'id':'receiver','kind':kind,'path':rel,'size_bytes':f.stat().st_size,'sha256':m.sha_file(f)})
  csvfile(base/'input_payload_hashes.csv',rows);csvfile(base/'donor_plan.csv',[{'id':'receiver'}])
  cp=base/'teacher.pth';cp.write_bytes(b'fixture not a tensor')
  writejson(base/'checkpoint_pin.json',{'checkpoint_sha256':m.sha_file(cp)})
  inp={'dataset_root':str(root),'splits':[],'checkpoint':{'target_path':str(cp),'bytes':cp.stat().st_size,'sha256':m.sha_file(cp)}};writejson(base/'inputs.json',inp)
  return base,root,cp
 def test_04_inputs_ID_layout_accepted_without_tensor_load(self):
  with tempfile.TemporaryDirectory() as td:
   b,root,cp=self.payload_fixture(td)
   with patch.object(m,'validate_lists',return_value={'val':{'receiver'}}):r=m.check_inputs(b)
   self.assertEqual(r['required_images'],1);self.assertFalse(r['checkpoint_tensor_load'])
 def test_05_missing_image_fails(self):
  with tempfile.TemporaryDirectory() as td:
   b,root,cp=self.payload_fixture(td);(root/'JPEGImages/receiver.jpg').unlink()
   with patch.object(m,'validate_lists',return_value={'val':{'receiver'}}),self.assertRaisesRegex(AssertionError,'MISSING_PAYLOAD'):m.check_inputs(b)
 def test_06_missing_mask_fails(self):
  with tempfile.TemporaryDirectory() as td:
   b,root,cp=self.payload_fixture(td);(root/'SegmentationClassAug/receiver.png').unlink()
   with patch.object(m,'validate_lists',return_value={'val':{'receiver'}}),self.assertRaisesRegex(AssertionError,'MISSING_PAYLOAD'):m.check_inputs(b)
 def test_07_checkpoint_mismatch_fails_before_tensor_load(self):
  with tempfile.TemporaryDirectory() as td:
   b,root,cp=self.payload_fixture(td);cp.write_bytes(b'changed')
   with patch.object(m,'validate_lists',return_value={'val':{'receiver'}}),self.assertRaisesRegex(AssertionError,'CHECKPOINT_CHECKSUM'):m.check_inputs(b)
 def test_08_payload_traversal_fails(self):
  with tempfile.TemporaryDirectory() as td:
   b,root,cp=self.payload_fixture(td);rows=list(csv.DictReader(open(b/'input_payload_hashes.csv')));rows[0]['path']='../escaped.jpg';csvfile(b/'input_payload_hashes.csv',rows)
   with patch.object(m,'validate_lists',return_value={'val':{'receiver'}}),self.assertRaisesRegex(AssertionError,'UNSAFE_PAYLOAD'):m.check_inputs(b)
 def list_fixture(self,td):
  b=Path(td);v=['receiver_'+str(i) for i in range(1024)];rows=[]
  for k,ids in [('662',['donor_a']),('1323',['donor_a','donor_b']),('2646',['donor_a','donor_b','donor_c']),('val',v)]:
   f=b/(k+'.txt');f.write_text('\n'.join(ids)+'\n');rows.append({'key':k,'target_path':str(f),'bytes':f.stat().st_size,'sha256':m.sha_file(f),'lines':len(ids)})
  csvfile(b/'receiver_plan.csv',[{'id':id_,'batch':i//16} for i,id_ in enumerate(v)])
  csvfile(b/'donor_plan.csv',[{'id':'donor_a','split':split,'batch':i//16} for split in ['662','1323','2646'] for i in range(1024)])
  writejson(b/'plan.json',{'unique_donors':{'662':1,'1323':1,'2646':1}})
  return b,{'splits':rows}
 def test_09_nested_lists_and_fixed_slots(self):
  with tempfile.TemporaryDirectory() as td:
   b,i=self.list_fixture(td);self.assertEqual(len(m.validate_lists(b,i)['val']),1024)
 def test_10_split_checksum_mismatch_fails(self):
  with tempfile.TemporaryDirectory() as td:
   b,i=self.list_fixture(td);(b/'662.txt').write_text('changed\n')
   with self.assertRaisesRegex(AssertionError,'SPLIT_CHECKSUM'):m.validate_lists(b,i)
 def test_11_preparation_binding_cannot_be_faked(self):
  with tempfile.TemporaryDirectory() as td:
   b=Path(td);writejson(b/'preparation.json',{'package_manifest_sha256':'incorrect'})
   with self.assertRaisesRegex(AssertionError,'PREPARATION_PACKAGE'):m.require_prepared(m.C,b)
 def test_12_real_model_action_wrong_host_fails(self):
  with patch('socket.gethostname',return_value='not-A6000'),self.assertRaisesRegex(AssertionError,'WRONG_HOST'):m.require_target()
 def test_13_package_extra_file_fails(self):
  with tempfile.TemporaryDirectory() as td:
   b=Path(td);writejson(b/'package_manifest.json',{});(b/'unexpected.txt').write_text('x')
   with self.assertRaisesRegex(AssertionError,'PACKAGE_FILE_SET'):m.verify_package(b)
 def test_14_no_new_augmentation_or_sampling_randomness_in_port(self):
  eq=m.load('scientific_equivalence.json');self.assertEqual(eq['science_changes'],[]);self.assertTrue(eq['receiver_and_donor_plans_byte_identical']);self.assertTrue(eq['recipe_byte_identical'])
 def test_16_transfer_member_safety(self):
  from transfer_inputs import safe_member
  self.assertTrue(safe_member('augseg_inputs/VOC2012/JPEGImages/id.jpg'))
  for n in ['/abs/path','../escaped','augseg_inputs/../../x','credentials/token']:
   self.assertFalse(safe_member(n))
 def test_15_missing_required_file_in_manifest_fails(self):
  with tempfile.TemporaryDirectory() as td:
   b,root,cp=self.payload_fixture(td);rows=list(csv.DictReader(open(b/'input_payload_hashes.csv')));csvfile(b/'input_payload_hashes.csv',rows[:1])
   with patch.object(m,'validate_lists',return_value={'val':{'receiver'}}),self.assertRaisesRegex(AssertionError,'INCOMPLETE_REQUIRED'):m.check_inputs(b)
if __name__=='__main__':
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PortableChecks))
 writejson(Path(os.environ['DIAGNOSTIC_OUTPUT'])/'portable_tests.json',{'passed':result.wasSuccessful(),'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors)})
 raise SystemExit(0 if result.wasSuccessful() else 1)
