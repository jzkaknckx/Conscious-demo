import unittest
import numpy as np
import torch
from nns.memorygraphs import graph_memorypool_onceoptimizer as g


class GradRefinementTests(unittest.TestCase):
    def test_parent_support_is_refined_not_copied(self):
        cfg=g.MemoryConfig();cfg.device=torch.device('cpu')
        shape=(16,16);mask=np.zeros(shape,bool);mask[8,2:10]=True
        points=np.flatnonzero(mask)
        parent=g.Region(0,0,1,points,points,[int(points[0])])
        parent.adjacency={int(p):[(int(q),1.) for q in points if abs(int(q)-int(p))==1] for p in points}
        report=g.BuildReport(1,shape,np.ones(shape,bool));report.regions=[parent]
        report.labels[0]=np.where(mask,0,-1)
        values=torch.full((1,1,*shape),.2);values[:,:,:,6:]=.9
        tangent=np.zeros((*shape,2));tangent[:,:,0]=1
        support=g.FeatureSupport(2,values,mask.copy(),np.ones(shape),mask.copy(),tangent,
            mask.copy(),np.zeros(shape,bool),[int(points[0])],torch.ones(1,1),.8,1)
        edge=g.FeatureSupport(0,values,mask.copy(),np.ones(shape),mask.copy(),tangent,
            mask.copy(),np.zeros(shape,bool),[int(points[0])],torch.ones(1,1),.8,1)
        assembler=g.GradRefinedRegionAssembler(cfg);assembler.bind_support(support,edge)
        regions,labels=assembler.assemble(support,[parent],report,10)
        self.assertEqual(len(regions),2)
        self.assertTrue(all(r.support_parent_region_id==0 and r.parent_region_id is None for r in regions))
        self.assertTrue(all(np.all(mask.flat[r.pixels]) for r in regions))
        self.assertNotEqual(labels[8,4],labels[8,8])

    def test_rgb_rectangle_geometry_is_not_gradient_anisotropy(self):
        mask=np.zeros((20,20),bool);mask[2:10,2:18]=True
        points=np.flatnonzero(mask)
        report=g.BuildReport(1,mask.shape,np.ones_like(mask))
        report.regions=[g.Region(0,1,2,points,points,[])]
        shape=g.region_shape_descriptors(report)[0]
        self.assertAlmostEqual(shape['axis_aligned_ratio'],2.)
        self.assertAlmostEqual(shape['pca_box_ratio'],2.)
        self.assertAlmostEqual(shape['moment_elongation'],2.)


if __name__=='__main__':unittest.main()
